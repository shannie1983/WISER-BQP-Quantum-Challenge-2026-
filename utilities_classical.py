"""
utilities_classical.py
======================
Classical utilities for NS-PINN: encodings, adaptive loss weighting,
the unified PINN model, and NS physics residuals.

Contents
--------
  SECTION 1  — Input encodings
               NoEncoding, FourierEncoding, AdaptiveFourierEncoding
  SECTION 2  — Adaptive loss weightings
               StaticWeighting, SoftAdaptWeighting, DynRatioWeighting,
               GradNormWeighting, CombinedWeighting
  SECTION 3  — UnifiedPINN model
               All classical PINN variants in one configurable class.
               Presets: mlp, pinn, fourier_pinn, hardbc_pinn,
                        softadapt_pinn, dynratio_pinn, rar_pinn,
                        best_classical, adaptive_combined, deep_rar, ...
  SECTION 4  — NS physics loss (classical autograd)
               compute_physics_loss(), unified_loss()
  SECTION 5  — Data loading helper
               load_ns_data()  — returns 7-feature split

Exports (public API)
--------------------
  from utilities_classical import (
      UnifiedPINN, build_unified, unified_loss,
      load_ns_data,
      NoEncoding, FourierEncoding, AdaptiveFourierEncoding,
      PRESETS, list_unified,
  )

Physics constants
-----------------
  NI  = 7   inputs: [x, t, p_ratio, mu, rho_L, rho_R, p_R]
  NO  = 3   outputs: [rho, u, p]
  PHYS_LAMS = {'mass':6.8e-3, 'mom':4.4e-3, 'ic':3.2e-3, 'bc':1.7e-3}
"""

from __future__ import annotations
import math, os, json, warnings
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
GAMMA=1.4; PR=0.72; R_GAS=1.0
CP=GAMMA*R_GAS/(GAMMA-1.); CV=R_GAS/(GAMMA-1.)

NI=7    # inputs:  x, t, p_ratio, mu, rho_L, rho_R, p_R
NO=3    # outputs: rho, u, p
H=128

PHYS_LAMS={'mass':6.8e-3,'mom':4.4e-3,'ic':3.2e-3,'bc':1.7e-3}
PDE_TERMS=list(PHYS_LAMS.keys())
RAMP_START=20; RAMP_END=50


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT ENCODINGS
# ══════════════════════════════════════════════════════════════════════════════

class NoEncoding(nn.Module):
    """Identity: passes NI=7 normalised inputs unchanged. out_dim=7."""
    def __init__(self): super().__init__(); self.out_dim=NI
    def forward(self,x): return x


class FourierEncoding(nn.Module):
    """
    Fixed random Fourier features: x → [sin(Bx), cos(Bx)] ∈ R^(2·n_freq).
    B ~ N(0,σ²) is fixed (not trained) — adaptive σ defeats the purpose.

    Parameters
    ----------
    n_freq : int   number of frequencies
    sigma  : float frequency scale (2.0 = good for shock tube)
    """
    def __init__(self,n_freq=64,sigma=2.0):
        super().__init__()
        B=torch.randn(NI,n_freq)*sigma
        self.register_buffer('B',B)
        self.out_dim=2*n_freq
    def forward(self,x):
        proj=x@self.B
        return torch.cat([torch.sin(proj),torch.cos(proj)],dim=-1)


class AdaptiveFourierEncoding(nn.Module):
    """
    Per-frequency learnable σ_k (log-space): B_eff[:,k] = B_base[:,k] × σ_k.
    Warmup schedule + optional SoftAdapt frequency boost.

    Parameters
    ----------
    n_freq         : int
    sigma_init     : float  initial σ for all frequencies
    sigma_min/max  : float  clamping bounds
    warmup_epochs  : int    σ ramps from 0.3 to 1.0 × sigma
    use_softadapt  : bool   boost underused frequencies
    """
    def __init__(self,n_freq=64,sigma_init=2.0,sigma_min=0.1,sigma_max=20.,
                 warmup_epochs=20,use_softadapt=True):
        super().__init__()
        self.n_freq=n_freq; self.sigma_min=sigma_min; self.sigma_max=sigma_max
        self.warmup_epochs=warmup_epochs; self.use_softadapt=use_softadapt
        self.out_dim=2*n_freq; self._epoch=0; self._usage_momentum=0.95
        self.register_buffer('B_base',torch.randn(NI,n_freq))
        self.log_sigma=nn.Parameter(torch.zeros(n_freq)+math.log(sigma_init))
        self.register_buffer('sigma_scale',torch.tensor(1.0))
        self.register_buffer('freq_usage',torch.ones(n_freq))
        self.register_buffer('freq_usage_prev',torch.ones(n_freq))
        self.freq_weights=nn.Parameter(torch.ones(2*n_freq))
    def _get_sigma(self):
        return torch.exp(self.log_sigma).clamp(self.sigma_min,self.sigma_max)*self.sigma_scale
    def forward(self,x):
        B=self.B_base*self._get_sigma().unsqueeze(0); proj=x@B
        z=torch.cat([torch.sin(proj),torch.cos(proj)],dim=-1)
        if self.training:
            with torch.no_grad():
                usage=z.abs().mean(0); up=(usage[:self.n_freq]+usage[self.n_freq:])/2
                self.freq_usage=self._usage_momentum*self.freq_usage+(1-self._usage_momentum)*up
        return z*torch.abs(self.freq_weights)
    def update(self,epoch):
        self._epoch=epoch
        if epoch<self.warmup_epochs: self.sigma_scale.fill_(0.3+0.7*epoch/max(self.warmup_epochs,1))
        else: self.sigma_scale.fill_(1.0)
        if self.use_softadapt and epoch>5 and epoch%5==0:
            with torch.no_grad():
                rate=(self.freq_usage-self.freq_usage_prev)/(self.freq_usage_prev+1e-8)
                boost=torch.exp(-rate/0.1); boost=boost/boost.mean()
                self.log_sigma.data+=0.01*(boost-1.)
                self.log_sigma.data.clamp_(math.log(self.sigma_min),math.log(self.sigma_max))
                self.freq_usage_prev.copy_(self.freq_usage)


def _make_encoding(enc,**kw):
    if enc=='none':             return NoEncoding()
    if enc=='fourier':          return FourierEncoding(kw.get('n_freq',64),kw.get('sigma',2.0))
    if enc=='adaptive_fourier': return AdaptiveFourierEncoding(
        kw.get('n_freq',64),kw.get('sigma_init',2.0),kw.get('sigma_min',0.1),
        kw.get('sigma_max',20.),kw.get('warmup_epochs',20),kw.get('enc_softadapt',True))
    raise ValueError(f"encoding={enc!r}. Choose: none, fourier, adaptive_fourier")


class _Sin(nn.Module):
    def forward(self,x): return torch.sin(x)

def _make_act(name):
    return {'tanh':nn.Tanh(),'sin':_Sin(),'relu':nn.ReLU(),'gelu':nn.GELU(),'swish':nn.SiLU()}.get(name) or (_ for _ in ()).throw(ValueError(f"activation={name!r}"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ADAPTIVE LOSS WEIGHTINGS
# ══════════════════════════════════════════════════════════════════════════════

class StaticWeighting:
    """Fixed PHYS_LAMS — no adaptation."""
    def __init__(self,lams=None): self.lams=dict(lams or PHYS_LAMS)
    def update(self,epoch,log): pass
    def get_lams(self): return dict(self.lams)
    def description(self): return f"static"


class SoftAdaptWeighting:
    """λ_k ← softmax(ΔL_k/L_k/T) × base. Boosts stagnating terms."""
    def __init__(self,T=0.1,update_freq=5,base_lam=1e-3):
        self.T=T; self.update_freq=update_freq; self.base_lam=base_lam
        self.lams={k:base_lam for k in PDE_TERMS}; self._prev={k:None for k in PDE_TERMS}
    def update(self,epoch,log):
        if epoch%self.update_freq!=0: return
        rates={}
        for k in PDE_TERMS:
            cur=log.get(k+'_raw',0.); prev=self._prev[k]
            rates[k]=(cur-prev)/prev if prev and prev>1e-10 and cur>0 else 0.
            self._prev[k]=cur
        rv=np.array([rates[k] for k in PDE_TERMS])
        w=np.exp(rv/self.T); w=w/w.sum()*len(PDE_TERMS)
        for i,k in enumerate(PDE_TERMS): self.lams[k]=float(np.clip(self.base_lam*w[i],1e-6,1.))
    def get_lams(self): return dict(self.lams)
    def description(self): return f"softadapt(T={self.T})"


class DynRatioWeighting:
    """λ_k ← target_pct × L_data / L_k. Keeps each term at fixed % of data loss."""
    def __init__(self,target_pct=0.03,update_freq=5,min_lam=1e-6,max_lam=1.):
        self.target_pct=target_pct; self.update_freq=update_freq
        self.min_lam=min_lam; self.max_lam=max_lam; self.lams=dict(PHYS_LAMS)
    def update(self,epoch,log):
        if epoch%self.update_freq!=0: return
        L=log.get('data',0.)
        if L<=0: return
        for k in PDE_TERMS:
            raw=log.get(k+'_raw',0.)
            if raw>0: self.lams[k]=float(np.clip(self.target_pct*L/raw,self.min_lam,self.max_lam))
    def get_lams(self): return dict(self.lams)
    def description(self): return f"dynratio({self.target_pct:.1%})"


class GradNormWeighting:
    """Balance gradient norms across terms (Chen et al. 2018)."""
    def __init__(self,alpha=1.5,lr_lam=0.025,update_freq=5):
        self.alpha=alpha; self.lr_lam=lr_lam; self.update_freq=update_freq
        self.lams={k:1. for k in PDE_TERMS}; self._L0={}; self._step=0
    def update(self,epoch,log):
        self._step+=1
        if self._step%self.update_freq!=0: return
        raw={k:log.get(k+'_raw',0.) for k in PDE_TERMS}
        if not self._L0: self._L0={k:max(float(v),1e-8) for k,v in raw.items()}
        G={k:float(raw[k])*self.lams[k] for k in PDE_TERMS}
        G_avg=np.mean(list(G.values()))+1e-8; r={k:float(raw[k])/self._L0[k] for k in PDE_TERMS}
        r_avg=np.mean(list(r.values()))+1e-8; total=0.
        for k in PDE_TERMS:
            tgt=G_avg*(r[k]/r_avg)**self.alpha
            self.lams[k]=max(1e-4,self.lams[k]-self.lr_lam*(G[k]-tgt)*np.sign(G[k]-tgt)); total+=self.lams[k]
        for k in PDE_TERMS: self.lams[k]=self.lams[k]*len(PDE_TERMS)/total
    def get_lams(self): return dict(self.lams)
    def description(self): return f"gradnorm(α={self.alpha})"


class CombinedWeighting:
    """α × SoftAdapt + (1-α) × DynRatio — handles both rate and magnitude."""
    def __init__(self,alpha=0.5,softadapt_kw=None,dynratio_kw=None):
        self.alpha=alpha; self._sa=SoftAdaptWeighting(**(softadapt_kw or {}))
        self._dr=DynRatioWeighting(**(dynratio_kw or {}))
    def update(self,epoch,log): self._sa.update(epoch,log); self._dr.update(epoch,log)
    def get_lams(self):
        sa=self._sa.get_lams(); dr=self._dr.get_lams()
        return {k:self.alpha*sa[k]+(1-self.alpha)*dr[k] for k in PDE_TERMS}
    def description(self): return f"combined(α={self.alpha})"


def _make_weighting(mode,**kw):
    if mode=='static':    return StaticWeighting(kw.get('lams'))
    if mode=='softadapt': return SoftAdaptWeighting(kw.get('T',0.1),kw.get('update_freq',5),kw.get('base_lam',1e-3))
    if mode=='dynratio':  return DynRatioWeighting(kw.get('target_pct',0.03),kw.get('update_freq',5),kw.get('min_lam',1e-6),kw.get('max_lam',1.))
    if mode=='gradnorm':  return GradNormWeighting(kw.get('gn_alpha',1.5),kw.get('gn_lr',0.025),kw.get('update_freq',5))
    if mode=='combined':
        return CombinedWeighting(kw.get('combine_alpha',0.5),
            softadapt_kw=dict(T=kw.get('T',0.1),update_freq=kw.get('update_freq',5),base_lam=kw.get('base_lam',1e-3)),
            dynratio_kw=dict(target_pct=kw.get('target_pct',0.03),update_freq=kw.get('update_freq',5),min_lam=kw.get('min_lam',1e-6),max_lam=kw.get('max_lam',1.)))
    raise ValueError(f"loss_mode={mode!r}. Choose: static,softadapt,dynratio,gradnorm,combined")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UNIFIED PINN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedPINN(nn.Module):
    """
    One class for all classical NS-PINN variants.

    Parameters
    ----------
    model_type  : 'mlp' | 'pinn'
    encoding    : 'none' | 'fourier' | 'adaptive_fourier'
    loss_mode   : 'static' | 'softadapt' | 'dynratio' | 'gradnorm' | 'combined'
    hard_bc     : bool  — analytic u=0 at walls
    use_rar     : bool  — residual-adaptive collocation
    depth       : int 2–6
    hidden      : int
    activation  : 'tanh' | 'sin' | 'relu' | 'gelu' | 'swish'
    use_energy  : bool  — include energy equation (experimental)
    """
    def __init__(self,model_type='pinn',encoding='none',loss_mode='static',
                 hard_bc=False,use_rar=False,depth=3,hidden=H,
                 activation='tanh',use_energy=False,**kwargs):
        super().__init__()
        if model_type not in('mlp','pinn'): raise ValueError(f"model_type={model_type!r}")
        if depth<2 or depth>6: raise ValueError(f"depth={depth} must be 2–6")
        if activation in('relu','gelu','swish') and model_type=='pinn':
            warnings.warn(f"activation={activation!r} not smooth → 2nd-order derivatives=0. Use tanh/sin.",UserWarning)
        self.model_type=model_type; self.use_physics=(model_type=='pinn')
        self.hard_bc=hard_bc; self.use_rar=use_rar; self.use_energy=use_energy
        self._depth=depth; self._hidden=hidden; self._kwargs=kwargs
        self.encoding_name=encoding
        self.encoding_module=_make_encoding(encoding,**kwargs)
        enc_dim=self.encoding_module.out_dim
        self._activation_name=activation
        layers=[nn.Linear(enc_dim,hidden),_make_act(activation)]
        for _ in range(depth-2): layers+=[nn.Linear(hidden,hidden),_make_act(activation)]
        layers+=[nn.Linear(hidden,NO)]
        self.net=nn.Sequential(*layers)
        if activation=='sin': self._init_siren()
        self._weighting=_make_weighting(loss_mode,**kwargs) if self.use_physics else StaticWeighting()
        self._n_pool=kwargs.get('n_pool',2000); self._rar_frac=kwargs.get('rar_frac',0.5)

    def _init_siren(self):
        omega=1.0
        for i,layer in enumerate(self.net):
            if isinstance(layer,nn.Linear):
                with torch.no_grad():
                    if i==0: nn.init.uniform_(layer.weight,-1/layer.in_features,1/layer.in_features)
                    else: nn.init.uniform_(layer.weight,-math.sqrt(6/layer.in_features)/omega,math.sqrt(6/layer.in_features)/omega)

    def forward(self,x):
        z=self.encoding_module(x); out=self.net(z)
        if self.hard_bc:
            Xm=getattr(self,'_Xm',None); Xs=getattr(self,'_Xs',None)
            if Xm is not None: x_phys=(x[:,0]*Xs[0]+Xm[0]).clamp(0.,1.)
            else: x_phys=torch.sigmoid(x[:,0])
            mask=(x_phys*(1-x_phys)).unsqueeze(1)
            out=torch.cat([out[:,:1],out[:,1:2]*mask*4,out[:,2:]],1)
        return out

    def update(self,epoch,log):
        """Call each epoch to update adaptive weights and encoding sigma."""
        self._weighting.update(epoch,log)
        if isinstance(self.encoding_module,AdaptiveFourierEncoding):
            self.encoding_module.update(epoch)

    def get_lam_overrides(self): return self._weighting.get_lams()

    def sample_collocation(self,n_col,p_range,mu_range,Xm,Xs):
        """RAR: sample collocation points near high-residual regions."""
        plo,phi=p_range; mlo,mhi=mu_range
        U_=lambda n,a,b: torch.rand(n)*(b-a)+a
        xp=U_(self._n_pool,0.,1.); tp=U_(self._n_pool,0.,.2)
        prp=U_(self._n_pool,plo,phi); mup=U_(self._n_pool,mlo,mhi)
        with torch.no_grad():
            inp=torch.stack([(xp-Xm[0])/Xs[0],(tp-Xm[1])/Xs[1],
                             (prp-Xm[2])/Xs[2],(mup-Xm[3])/Xs[3]]+
                            [torch.zeros(self._n_pool)]*3,1)
            out=self.forward(inp)
            Ys=getattr(self,'_Ys',torch.ones(NO)); Ym=getattr(self,'_Ym',torch.zeros(NO))
            rho_=out[:,0]*Ys[0]+Ym[0]; u_=out[:,1]*Ys[1]+Ym[1]
            proxy=(rho_*u_-(rho_*u_).mean()).abs()
        n_high=int(n_col*self._rar_frac); n_rand=n_col-n_high
        _,top_idx=torch.topk(proxy,n_high); rand_idx=torch.randperm(self._n_pool)[:n_rand]
        sel=torch.cat([top_idx,rand_idx])
        return xp[sel],tp[sel],prp[sel],mup[sel]

    def describe(self):
        n=sum(p.numel() for p in self.parameters())
        print(f"\nUnifiedPINN: {self.model_type} | encoding={self.encoding_name}"
              f" | depth={self._depth} | hidden={self._hidden}"
              f" | hard_bc={self.hard_bc} | use_rar={self.use_rar}"
              f" | loss={self._weighting.description()} | params={n:,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NS PHYSICS LOSS (CLASSICAL AUTOGRAD)
# ══════════════════════════════════════════════════════════════════════════════

def _prim_grads(model,xc,tc,prc,muc,Xm,Xs,rho_Lc=None,rho_Rc=None,p_Rc=None):
    """Primitive variables + gradients via autograd. Supports 7-feature input."""
    dev=Xm.device; n=len(xc)
    def norm(v,i): return (v-Xm[i])/Xs[i]
    def fill(val,i): return torch.full((n,),float(Xm[i]),device=dev) if val is None else val
    rLn=(fill(rho_Lc,4)-Xm[4])/Xs[4]; rRn=(fill(rho_Rc,5)-Xm[5])/Xs[5]
    pRn=(fill(p_Rc,6)-Xm[6])/Xs[6]
    inp=torch.stack([norm(xc,0),norm(tc,1),norm(prc,2),norm(muc,3),rLn,rRn,pRn],1).requires_grad_(True)
    out=model(inp)
    Ym=getattr(model,'_Ym',torch.zeros(NO,device=dev)); Ys=getattr(model,'_Ys',torch.ones(NO,device=dev))
    grads=[torch.autograd.grad(out[:,k].sum(),inp,create_graph=True,retain_graph=True)[0] for k in range(NO)]
    def pdx(k): return grads[k][:,0]/Xs[0]*Ys[k]
    def pdt(k): return grads[k][:,1]/Xs[1]*Ys[k]
    rho=out[:,0]*Ys[0]+Ym[0]; u=out[:,1]*Ys[1]+Ym[1]; p=out[:,2]*Ys[2]+Ym[2]
    d2u=torch.autograd.grad(pdx(1).sum(),inp,create_graph=True,retain_graph=True)[0][:,0]/Xs[0]*Ys[1]
    return rho,u,p,pdt(0),pdt(1),pdx(0),pdx(1),pdx(2),d2u


def compute_physics_loss(model,xb,Xm,Xs,p_range,mu_range,n_col=32,n_ic=16,n_bc=8):
    """
    Classical NS residuals via autograd.
    xb shape: (B, 7) — uses cols 3..6 for IC parameters.
    Returns dict: mass, mom, ic, bc
    """
    plo,phi=p_range; mlo,mhi=mu_range; dev=Xm.device
    U_=lambda n,a,b: torch.rand(n,device=dev)*(b-a)+a
    mu_p=xb[:,3].detach()*Xs[3]+Xm[3]
    rL_p=xb[:,4].detach()*Xs[4]+Xm[4] if xb.shape[1]>4 else None
    rR_p=xb[:,5].detach()*Xs[5]+Xm[5] if xb.shape[1]>5 else None
    pR_p=xb[:,6].detach()*Xs[6]+Xm[6] if xb.shape[1]>6 else None
    def smu(n): return mu_p[:n].clone() if len(mu_p)>=n else U_(n,mlo,mhi)
    def srL(n): return rL_p[:n].clone() if rL_p is not None and len(rL_p)>=n else torch.full((n,),float(Xm[4]),device=dev)
    def srR(n): return rR_p[:n].clone() if rR_p is not None and len(rR_p)>=n else torch.full((n,),float(Xm[5]),device=dev)
    def spR(n): return pR_p[:n].clone() if pR_p is not None and len(pR_p)>=n else torch.full((n,),float(Xm[6]),device=dev)
    comp={}
    # Collocation
    xc=U_(n_col,0.,1.); tc=U_(n_col,0.,.2); prc=U_(n_col,plo,phi)
    muc=smu(n_col); rLc=srL(n_col); rRc=srR(n_col); pRc=spR(n_col)
    rho,u,p,drho_dt,du_dt,drho_dx,du_dx,dp_dx,d2u=_prim_grads(model,xc,tc,prc,muc,Xm,Xs,rLc,rRc,pRc)
    comp['mass']=((drho_dt+u.detach()*drho_dx+rho.detach()*du_dx)**2).mean()
    comp['mom'] =((rho.detach()*(du_dt+u.detach()*du_dx)+dp_dx-(4./3.)*muc*d2u)**2).mean()
    # IC
    xi=U_(n_ic,0.,1.); pri=U_(n_ic,plo,phi); mui=smu(n_ic); ti=torch.zeros(n_ic,device=dev)
    rLi=srL(n_ic); rRi=srR(n_ic); pRi=spR(n_ic)
    ii=torch.stack([(xi-Xm[0])/Xs[0],(ti-Xm[1])/Xs[1],(pri-Xm[2])/Xs[2],(mui-Xm[3])/Xs[3],
                    (rLi-Xm[4])/Xs[4],(rRi-Xm[5])/Xs[5],(pRi-Xm[6])/Xs[6]],1)
    oi=model(ii)
    Ym=getattr(model,'_Ym',torch.zeros(NO,device=dev)); Ys=getattr(model,'_Ys',torch.ones(NO,device=dev))
    ri=oi[:,0]*Ys[0]+Ym[0]; ui_=oi[:,1]*Ys[1]+Ym[1]; pi_=oi[:,2]*Ys[2]+Ym[2]
    comp['ic']=(((ri-torch.where(xi<.5,rLi,rRi))**2)+ui_**2+((pi_-torch.where(xi<.5,pri*pRi,pRi))**2)).mean()
    # BC
    tb=U_(n_bc,0.,.2); prb=U_(n_bc,plo,phi); mub=smu(n_bc)
    rLb=srL(n_bc); rRb=srR(n_bc); pRb=spR(n_bc)
    bcl=torch.tensor(0.,device=dev)
    for xw_val in [0.,1.]:
        xw=torch.full((n_bc,),xw_val,device=dev)
        _,_,_,_,_,dr,du2,dp2,_=_prim_grads(model,xw,tb,prb,mub,Xm,Xs,rLb,rRb,pRb)
        bcl=bcl+(dr**2+du2**2+dp2**2).mean()
    comp['bc']=bcl
    return comp


def unified_loss(model,xb,yb,epoch,Xm,Xs,p_range,mu_range,
                 ramp_start=RAMP_START,ramp_end=RAMP_END,n_col=32,n_ic=16,n_bc=8):
    """
    Total loss = L_data + ramp × Σ_k λ_k × L_k.

    Returns (loss_tensor, log_dict).
    log_dict keys: data, mass_raw, mom_raw, ic_raw, bc_raw,
                   mass_weighted, ..., ramp, total_physics
    """
    pred=model(xb); mse=nn.functional.mse_loss(pred,yb)
    log={'data':float(mse.detach())}; loss=mse
    r=max(0.,min(1.,(epoch-ramp_start)/max(ramp_end-ramp_start,1)))
    if not(getattr(model,'use_physics',False) and r>0):
        log.update({k+'_raw':0. for k in PDE_TERMS}); log.update({k+'_weighted':0. for k in PDE_TERMS})
        log['ramp']=r; return loss,log
    comp=compute_physics_loss(model,xb,Xm,Xs,p_range,mu_range,n_col,n_ic,n_bc)
    lams=model.get_lam_overrides()
    for k in PDE_TERMS:
        if k not in comp: continue
        wt=r*lams.get(k,PHYS_LAMS.get(k,1e-3))*comp[k]; loss=loss+wt
        log[k+'_raw']=float(comp[k].detach()); log[k+'_weighted']=float(wt.detach())
    log['ramp']=r; log['total_physics']=sum(log.get(k+'_weighted',0.) for k in PDE_TERMS)
    return loss,log


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_ns_data(data_dir,n_scenarios=70,t_stride=2,train_frac=0.80,val_frac=0.10,seed=0):
    """
    Load NS shock-tube data and build 7-feature normalised tensors.

    Input features (NI=7):
      [x, t, p_ratio, mu, rho_L, rho_R, p_R]

    Returns dict: X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                  Xm, Xs, Ym, Ys, p_range, mu_range,
                  n_samples, feature_names, output_names
    """
    idx_path=os.path.join(data_dir,'index.json')
    index=json.load(open(idx_path))
    scenarios=index['scenarios'][:n_scenarios]
    Xl,Yl,pvals,muvals=[],[],[],[]
    for sc in scenarios:
        f=np.load(os.path.join(data_dir,sc['filename']))
        x=f['x']; N=len(x); pr=float(sc['p_ratio']); mu=float(sc['mu'])
        rL=float(f['rho_L']); rR=float(f['rho_R']); pR=float(f['p_R'])
        pvals.append(pr); muvals.append(mu)
        for si in range(0,len(f['t_snaps']),t_stride):
            row=np.stack([x,np.full(N,float(f['t_snaps'][si])),np.full(N,pr),np.full(N,mu),
                          np.full(N,rL),np.full(N,rR),np.full(N,pR)],1).astype(np.float32)
            Xl.append(row); Yl.append(np.stack([f['rho'][si],f['u'][si],f['p'][si]],1).astype(np.float32))
    X_all=np.concatenate(Xl); Y_all=np.concatenate(Yl)
    Xm_np=X_all.mean(0); Xs_np=X_all.std(0).clip(1e-6)
    Ym_np=Y_all.mean(0); Ys_np=Y_all.std(0).clip(1e-6)
    Xn=((X_all-Xm_np)/Xs_np).astype(np.float32); Yn=((Y_all-Ym_np)/Ys_np).astype(np.float32)
    rng=np.random.RandomState(seed); idx=rng.permutation(len(Xn))
    n_tr=int(train_frac*len(Xn)); n_va=int(val_frac*len(Xn))
    tr_i,va_i,te_i=idx[:n_tr],idx[n_tr:n_tr+n_va],idx[n_tr+n_va:]
    def T(a,i): return torch.tensor(a[i],dtype=torch.float32)
    return dict(
        X_tr=T(Xn,tr_i),Y_tr=T(Yn,tr_i),X_va=T(Xn,va_i),Y_va=T(Yn,va_i),
        X_te=T(Xn,te_i),Y_te=T(Yn,te_i),
        Xm=torch.tensor(Xm_np,dtype=torch.float32),Xs=torch.tensor(Xs_np,dtype=torch.float32),
        Ym=torch.tensor(Ym_np,dtype=torch.float32),Ys=torch.tensor(Ys_np,dtype=torch.float32),
        p_range=(float(min(pvals)),float(max(pvals))),mu_range=(float(min(muvals)),float(max(muvals))),
        n_samples=len(X_all),feature_names=['x','t','p_ratio','mu','rho_L','rho_R','p_R'],
        output_names=['rho','u','p'])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FACTORY AND PRESETS
# ══════════════════════════════════════════════════════════════════════════════

PRESETS={
    'mlp':dict(model_type='mlp',encoding='none',loss_mode='static',hard_bc=False,use_rar=False,depth=3,hidden=128,activation='tanh',desc="MLP baseline R²≈0.998"),
    'pinn':dict(model_type='pinn',encoding='none',loss_mode='static',hard_bc=False,use_rar=False,depth=3,hidden=128,activation='tanh',desc="PINN static-λ R²≈0.980"),
    'fourier_pinn':dict(model_type='pinn',encoding='fourier',loss_mode='static',depth=3,hidden=128,activation='tanh',n_freq=64,sigma=2.0,desc="FourierPINN R²≈0.989"),
    'hardbc_pinn':dict(model_type='pinn',encoding='none',loss_mode='static',hard_bc=True,depth=3,hidden=128,activation='tanh',desc="HardBC PINN R²≈0.984"),
    'softadapt_pinn':dict(model_type='pinn',encoding='none',loss_mode='softadapt',depth=3,hidden=128,activation='tanh',T=0.1,update_freq=5,desc="SoftAdapt PINN R²≈0.980"),
    'dynratio_pinn':dict(model_type='pinn',encoding='none',loss_mode='dynratio',depth=3,hidden=128,activation='tanh',target_pct=0.03,desc="DynRatio PINN R²≈0.980"),
    'rar_pinn':dict(model_type='pinn',encoding='none',loss_mode='static',use_rar=True,depth=3,hidden=128,activation='tanh',n_pool=2000,rar_frac=0.5,desc="RAR PINN R²≈0.988"),
    'siren_pinn':dict(model_type='pinn',encoding='none',loss_mode='static',depth=3,hidden=128,activation='sin',desc="SIREN PINN (use sin activation, ω=1)"),
    'best_classical':dict(model_type='pinn',encoding='adaptive_fourier',loss_mode='combined',hard_bc=True,use_rar=True,depth=4,hidden=128,activation='tanh',n_freq=64,sigma_init=2.0,combine_alpha=0.5,target_pct=0.03,T=0.1,update_freq=5,n_pool=2000,rar_frac=0.5,desc="Best classical: AdaptFourier+Combined+HardBC+RAR"),
    'fourier_softadapt':dict(model_type='pinn',encoding='fourier',loss_mode='softadapt',depth=3,hidden=128,activation='tanh',n_freq=64,sigma=2.0,T=0.1,desc="Fourier+SoftAdapt"),
    'adaptive_combined':dict(model_type='pinn',encoding='adaptive_fourier',loss_mode='combined',depth=4,hidden=128,activation='tanh',n_freq=64,combine_alpha=0.5,desc="AdaptFourier+Combined"),
    'deep_rar':dict(model_type='pinn',encoding='fourier',loss_mode='softadapt',hard_bc=True,use_rar=True,depth=5,hidden=128,activation='tanh',n_freq=64,sigma=2.0,n_pool=2000,rar_frac=0.6,desc="Deep Fourier+SoftAdapt+HardBC+RAR"),
}


def build_unified(name='pinn',**overrides):
    """Build UnifiedPINN from preset name with optional overrides."""
    if name=='custom': cfg={}
    elif name in PRESETS: cfg={k:v for k,v in PRESETS[name].items() if k!='desc'}
    else: raise ValueError(f"Unknown preset {name!r}. Available: {list(PRESETS)} or 'custom'")
    cfg.update(overrides)
    return UnifiedPINN(**cfg)


def list_unified(verbose=True):
    """Print table of all presets."""
    rows=[]
    for name,cfg in PRESETS.items():
        kw={k:v for k,v in cfg.items() if k!='desc'}; desc=cfg.get('desc','')
        try: m=UnifiedPINN(**kw); n=sum(p.numel() for p in m.parameters())
        except Exception as e: n=0; desc=f"ERROR: {e}"
        rows.append((name,n,desc))
    if verbose:
        print(f"\n{'='*70}\nUnifiedPINN Presets\n{'─'*70}")
        print(f"  {'Name':<24} {'Params':>8}  Description")
        print(f"{'─'*70}")
        for name,n,desc in rows: print(f"  {name:<24} {n:>8,}  {desc}")
        print(f"{'='*70}")
    return rows


if __name__=='__main__':
    print("utilities_classical.py — self test\n")
    list_unified()
    x=torch.rand(4,NI); y=torch.rand(4,NO)
    Xm=torch.zeros(NI); Xs=torch.ones(NI); p_range=(2.,10.); mu_range=(0.,0.02)
    for name in ['mlp','pinn','fourier_pinn','softadapt_pinn','rar_pinn','best_classical']:
        m=build_unified(name); m._Xm=Xm; m._Xs=Xs; m._Ym=torch.zeros(NO); m._Ys=torch.ones(NO)
        loss,log=unified_loss(m,x,y,60,Xm,Xs,p_range,mu_range,n_col=8,n_ic=4,n_bc=4)
        n=sum(p.numel() for p in m.parameters())
        print(f"  ✓ {name:<22} params={n:>7,}  data={log['data']:.4f}  phys={log.get('total_physics',0.):.5f}")
    print("Done ✓")
