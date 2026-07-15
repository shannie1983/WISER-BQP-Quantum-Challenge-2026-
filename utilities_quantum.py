"""
utilities_quantum.py
====================
Quantum utilities for NS-PINN: encodings, VQC ansatz, LCU loss,
quantum regularisation, and the ScalableQPINN model.

Contents
--------
  SECTION 1  — Bloch-space primitives
  SECTION 2  — LCU block-encoding PDE loss  (from vqc_builder.py)
               LCUChannel, NSQuantumPDELoss
  SECTION 3  — Quantum regularisation loss  (from vqc_builder.py)
               QuantumFidelityLoss, QuantumLoss
  SECTION 4  — Qiskit circuit builders      (from vqc_builder.py)
               8 encodings × 8 ansatz; build_vqc(), build_torch_vqc()
  SECTION 5  — Native PyTorch statevector VQC  (from ns_models.py)
               ScalableQPINN, ScalableQAPINN
               gate primitives: _rx, _ry, _rz, _crz, _cry, _zz, ...
  SECTION 6  — Quantum forward helpers
               forward_with_raw()  — returns (pred, raw_Z) together
               get_device(), model_to_device(), estimate_vram()
  SECTION 7  — Qubit / preset utilities
               QPINN_PRESETS, qubit_info(), vram_table()

Exports (public API)
--------------------
  # LCU loss
  from utilities_quantum import NSQuantumPDELoss, LCUChannel

  # Quantum regularisation
  from utilities_quantum import QuantumLoss, QuantumFidelityLoss

  # Native PyTorch quantum models
  from utilities_quantum import ScalableQPINN, ScalableQAPINN

  # Qiskit-based VQC
  from utilities_quantum import build_vqc, build_torch_vqc
  from utilities_quantum import ENCODING_OPTIONS, ANSATZ_OPTIONS

  # Forward helpers
  from utilities_quantum import forward_with_raw

  # Device / memory
  from utilities_quantum import get_device, model_to_device, estimate_vram, vram_table

  # Presets
  from utilities_quantum import QPINN_PRESETS, qubit_info
"""

from __future__ import annotations
import math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn

# ── Physical constants ────────────────────────────────────────────────────────
NI = 7   # inputs: x, t, p_ratio, mu, rho_L, rho_R, p_R
NO = 3   # outputs: rho, u, p

NQ_MIN, NQ_MAX = 2, 8
NL_MIN, NL_MAX = 1, 20
NQ_MIN_FULL    = NI   # min qubits for full feature encoding (=7); default is 8


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BLOCH-SPACE PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _bloch_fidelity(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Product-state fidelity from Z Bloch vectors. F = Π_i (1+aᵢbᵢ)/2."""
    return ((1.0 + z_a * z_b) / 2.0).prod(dim=-1)


def _encode_to_bloch(values: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Classical values → Bloch Z ∈ (-1,+1) via tanh, tiled to n_qubits."""
    B, d = values.shape
    if d >= n_qubits:
        v = values[:, :n_qubits]
    else:
        repeats = (n_qubits + d - 1) // d
        v = values.repeat(1, repeats)[:, :n_qubits]
    return torch.tanh(v)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LCU BLOCK-ENCODING PDE LOSS  (from vqc_builder.py)
# ══════════════════════════════════════════════════════════════════════════════

EQ_MASS=0; EQ_MOMENTUM=1; EQ_ENERGY=2; EQ_FIDELITY=3

_NS_PAULI_TERMS = {
    EQ_MASS: [
        ('III','identity baseline'),('ZII','ρ—∂ρ/∂t'),
        ('IZI','u—velocity'),('ZZI','ρu—mass flux'),
    ],
    EQ_MOMENTUM: [
        ('III','identity'),('ZZI','ρu—∂(ρu)/∂t'),
        ('IIZ','p—pressure grad'),('ZIZ','ρu²+p—convective'),
        ('IZI','ν·(-2u)—viscous diag'),('ZZI','ν·(ρu)—visc ρ'),
        ('IZZ','ν·(up)—visc u·p'),('ZZZ','ν·(ρup)—visc 3b'),
    ],
    EQ_ENERGY: [
        ('III','identity'),('ZZZ','E—∂E/∂t'),
        ('IZZ','(E+p)u—energy flux'),('ZIZ','ρu—kinetic'),
        ('IIZ','κ·p—thermal press'),('ZII','κ·ρ—thermal dens'),
        ('ZIZ','κ·(ρp)—thermal ρT'),('IZZ','κ·(up)—thermal vel'),
        ('IZI','ν·u—visc work'),('ZZI','ν·(ρu)—visc mom'),
        ('IZZ','ν·(up)—visc pres'),('ZZZ','ν·(ρup)—visc 3b'),
        ('ZII','νκ·ρ—vt ρ'),('IIZ','νκ·p—vt p'),
        ('ZZI','νκ·ρu—vt mom'),('ZIZ','νκ·ρp—vt full'),
    ],
    EQ_FIDELITY: [
        ('III','identity'),('ZII','Z on ρ'),
        ('IZI','Z on u'),('IIZ','Z on p'),
    ],
}
_EQ_N_ANC={EQ_MASS:2,EQ_MOMENTUM:3,EQ_ENERGY:4,EQ_FIDELITY:2}
_MOMENTUM_RHS=list(range(4,8))
_ENERGY_KAPPA=list(range(4,8)); _ENERGY_NU=list(range(8,12)); _ENERGY_NUKU=list(range(12,16))


def _scale_pauli_str(s: str, n: int) -> str:
    if len(s) >= n: return s[:n]
    return 'I'*(n-len(s))+s


class LCUChannel(nn.Module):
    """
    Single LCU channel: PREP → SELECT → PREP† for one NS equation.
    H_k = Σᵢ αᵢ Uᵢ  where αᵢ = softmax(raw_alpha).
    """
    def __init__(self, eq_id: int, n_qubits: int):
        super().__init__()
        self.eq_id=eq_id; self.n_qubits=n_qubits
        self.n_anc=_EQ_N_ANC[eq_id]; self.n_terms=2**self.n_anc
        self.raw_alpha=nn.Parameter(torch.zeros(self.n_terms))
        base=_NS_PAULI_TERMS[eq_id]
        while len(base)<self.n_terms: base=base+[('I'*3,'padding')]
        self.pauli_strings=[_scale_pauli_str(s,n_qubits) for s,_ in base[:self.n_terms]]
        self.pauli_descs=[d for _,d in base[:self.n_terms]]

    @property
    def alpha(self): return torch.softmax(self.raw_alpha,0)

    def _effective_alpha(self,nu,kappa):
        alpha=self.alpha
        scale=torch.ones(self.n_terms,device=alpha.device,dtype=alpha.dtype)
        if self.eq_id==EQ_MOMENTUM:
            for i in _MOMENTUM_RHS:
                if i<self.n_terms: scale[i]=nu
        elif self.eq_id==EQ_ENERGY:
            for i in _ENERGY_KAPPA:
                if i<self.n_terms: scale[i]=kappa
            for i in _ENERGY_NU:
                if i<self.n_terms: scale[i]=nu
            for i in _ENERGY_NUKU:
                if i<self.n_terms: scale[i]=nu*kappa
        return alpha*scale

    def _sign_matrix(self,device,dtype):
        signs=torch.ones(self.n_terms,self.n_qubits,device=device,dtype=dtype)
        for i,ps in enumerate(self.pauli_strings):
            for k,p in enumerate(ps[::-1]):
                if k>=self.n_qubits: break
                if p in('X','Y'): signs[i,k]=-1.
        return signs

    def _pauli_expectation(self,pauli_str,bloch_z):
        res=torch.ones(bloch_z.shape[0],device=bloch_z.device,dtype=bloch_z.dtype)
        for k,p in enumerate(pauli_str[::-1]):
            if k>=self.n_qubits: break
            if p=='Z': res=res*bloch_z[:,k]
            elif p in('X','Y'): return torch.zeros_like(res)
        return res

    def block_encode(self,z,nu=1.,kappa=1.):
        alpha=self._effective_alpha(nu,kappa)
        signs=self._sign_matrix(z.device,z.dtype)
        return torch.einsum('i,ij,bj->bj',alpha,signs,z)

    def block_decode(self,z_enc,nu=1.,kappa=1.):
        return self.block_encode(z_enc,nu,kappa)   # Paulis self-adjoint

    def reconstruction_loss(self,z,nu=1.,kappa=1.):
        return ((self.block_decode(self.block_encode(z,nu,kappa),nu,kappa)-z)**2).mean()

    def expectation(self,bloch_z,nu=1.,kappa=1.):
        alpha=self._effective_alpha(nu,kappa)
        exp_val=torch.zeros(bloch_z.shape[0],device=bloch_z.device,dtype=bloch_z.dtype)
        for i,ps in enumerate(self.pauli_strings):
            exp_val=exp_val+alpha[i]*self._pauli_expectation(ps,bloch_z)
        return exp_val

    def physical_loss(self,bloch_z,nu=1.,kappa=1.):
        return (self.expectation(bloch_z,nu,kappa)**2).mean()

    def fidelity_loss(self,bloch_z,z_y,nu=1.,kappa=1.):
        ze_y=self.block_encode(z_y,nu,kappa)
        ze_p=self.block_encode(bloch_z,nu,kappa)
        Fc=_bloch_fidelity(ze_y,ze_p); Fy=_bloch_fidelity(ze_y,ze_y)
        Fp=_bloch_fidelity(ze_p,ze_p)
        F_norm=(Fc/(Fy*Fp).clamp(1e-8).sqrt()).clamp(0.,1.)
        return (1.-F_norm).mean()


class NSQuantumPDELoss(nn.Module):
    """
    Full NS quantum PDE loss via LCU block-encoding.
    4 channels: mass / momentum / energy / fidelity.
    Each channel: PREP→SELECT→PREP† with trainable Pauli coefficients α_k.
    Trainable: raw_alpha per channel, log_nu, log_kappa, raw_lambda, raw_beta.

    Parameters
    ----------
    n_qubits   : int   — must match VQC raw output width
    mode       : str   — 'separate' | 'single'
    w_phys     : float — weight on ⟨ψ|H_k|ψ⟩² term
    w_fid      : float — weight on fidelity term
    w_recon    : float — weight on block-encoding roundtrip loss
    init_nu    : float — initial kinematic viscosity (m²/s)
    init_kappa : float — initial thermal conductivity (W/m·K)
    """
    EQ_NAMES=['mass','momentum','energy','fidelity']

    def __init__(self,n_qubits=8,mode='separate',
                 w_phys=0.1,w_fid=0.01,w_recon=0.01,
                 init_nu=1.8e-5,init_kappa=0.026):
        super().__init__()
        assert mode in('separate','single')
        self.n_qubits=n_qubits; self.mode=mode
        self.w_phys=w_phys; self.w_fid=w_fid; self.w_recon=w_recon
        self.log_nu   =nn.Parameter(torch.tensor(math.log(init_nu)))
        self.log_kappa=nn.Parameter(torch.tensor(math.log(init_kappa)))
        self.channels =nn.ModuleList([LCUChannel(k,n_qubits) for k in range(4)])
        self.raw_lambda=nn.Parameter(torch.zeros(3))
        self.raw_beta  =nn.Parameter(torch.zeros(3))

    @property
    def nu(self): return torch.exp(self.log_nu)
    @property
    def kappa(self): return torch.exp(self.log_kappa)
    @property
    def lambda_weights(self): return torch.softmax(self.raw_lambda,0)
    @property
    def beta_weights(self):   return torch.softmax(self.raw_beta,0)

    def _build_z_y_list(self,y_true,nu,kappa):
        rho=y_true[:,0:1]; u=y_true[:,1:2]; p=y_true[:,2:3]
        nd=nu.detach(); kd=kappa.detach(); nq=self.n_qubits
        def raw(v): return _encode_to_bloch(v,nq)
        z_raw=[
            raw(torch.cat([rho,u,rho],1)),
            raw(torch.cat([rho*u+nd*u,p,rho*u],1)),
            raw(torch.cat([u**2,p,p*u+kd*(p/(rho.abs()+1e-8))],1)),
            raw(y_true),
        ]
        return [self.channels[k].block_encode(z_raw[k],nu,kappa) for k in range(4)]

    def forward(self,qc_out,y_true=None):
        nu,kappa=self.nu,self.kappa
        z_y_list=self._build_z_y_list(y_true,nu,kappa) if y_true is not None else [None]*4
        if self.mode=='separate': loss=self._fwd_separate(qc_out,z_y_list,nu,kappa)
        else:                     loss=self._fwd_single(qc_out,z_y_list,nu,kappa)
        if y_true is not None:
            rho=y_true[:,0:1]; u=y_true[:,1:2]; p=y_true[:,2:3]
            nd=nu.detach(); kd=kappa.detach(); nq=self.n_qubits
            def raw(v): return _encode_to_bloch(v,nq)
            z_raws=[raw(torch.cat([rho,u,rho],1)),
                    raw(torch.cat([rho*u+nd*u,p,rho*u],1)),
                    raw(torch.cat([u**2,p,p*u+kd*(p/(rho.abs()+1e-8))],1)),
                    raw(y_true)]
            for k in range(4):
                loss=loss+self.w_recon*self.channels[k].reconstruction_loss(z_raws[k],nu,kappa)
        return loss

    def _fwd_separate(self,qc_out,z_y_list,nu,kappa):
        lam=self.lambda_weights
        total=torch.tensor(0.,device=qc_out.device)
        for k in range(3):
            ch=self.channels[k]
            Lp=ch.physical_loss(qc_out,nu,kappa)
            Lf=ch.fidelity_loss(qc_out,z_y_list[k],nu,kappa) if z_y_list[k] is not None else torch.tensor(0.,device=qc_out.device)
            total=total+lam[k]*(self.w_phys*Lp+self.w_fid*Lf)
        if z_y_list[3] is not None:
            total=total+self.w_fid*self.channels[3].fidelity_loss(qc_out,z_y_list[3])
        return total

    def _fwd_single(self,qc_out,z_y_list,nu,kappa):
        beta=self.beta_weights
        exp_total=sum(beta[k]*self.channels[k].expectation(qc_out,nu,kappa) for k in range(3))
        Lp=(exp_total**2).mean()
        Lf=torch.tensor(0.,device=qc_out.device)
        for k in range(3):
            if z_y_list[k] is not None:
                Lf=Lf+beta[k]*self.channels[k].fidelity_loss(qc_out,z_y_list[k],nu,kappa)
        if z_y_list[3] is not None:
            Lf=Lf+self.channels[3].fidelity_loss(qc_out,z_y_list[3])
        return self.w_phys*Lp+self.w_fid*Lf

    def breakdown(self,qc_out,y_true=None):
        nu,kappa=self.nu,self.kappa
        z_y_list=self._build_z_y_list(y_true,nu,kappa) if y_true is not None else [None]*4
        result={'mode':self.mode,'nu':nu.item(),'kappa':kappa.item(),
                'lambda_weights':self.lambda_weights.detach().tolist(),
                'beta_weights':self.beta_weights.detach().tolist()}
        total=torch.tensor(0.,device=qc_out.device)
        for k in range(3):
            ch=self.channels[k]
            Lp=ch.physical_loss(qc_out,nu,kappa)
            Lf=ch.fidelity_loss(qc_out,z_y_list[k],nu,kappa) if z_y_list[k] is not None else torch.tensor(0.)
            result[self.EQ_NAMES[k]]={'L_phys':Lp.item(),'L_fid':Lf.item(),'n_anc':ch.n_anc}
            total=total+self.lambda_weights[k]*(self.w_phys*Lp+self.w_fid*Lf)
        ch3=self.channels[3]
        Lf3=ch3.fidelity_loss(qc_out,z_y_list[3]) if z_y_list[3] is not None else torch.tensor(0.)
        result['fidelity']={'L_fid':Lf3.item(),'n_anc':ch3.n_anc}
        result['total']=(total+self.w_fid*Lf3).item()
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QUANTUM REGULARISATION LOSS  (from vqc_builder.py)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumFidelityLoss(nn.Module):
    """Supervised fidelity: 1 - F_norm(tanh(y_true), raw_out)."""
    def __init__(self,n_qubits,weight=0.01):
        super().__init__(); self.n_qubits=n_qubits; self.weight=weight
    def forward(self,qc_out,y_true):
        psi_y=_encode_to_bloch(y_true,self.n_qubits); psi_p=qc_out
        Fc=_bloch_fidelity(psi_y,psi_p); Fy=_bloch_fidelity(psi_y,psi_y); Fp=_bloch_fidelity(psi_p,psi_p)
        F_norm=(Fc/(Fy*Fp).clamp(1e-8).sqrt()).clamp(0.,1.)
        return self.weight*(1.-F_norm).mean()


class QuantumLoss(nn.Module):
    """
    Quantum regularisation: barren plateau + concentration + entanglement + fidelity.
    Computed purely from raw VQC ⟨Z⟩ outputs — no Qiskit needed.
    """
    def __init__(self,n_qubits=3,alpha=0.01,beta=10.,gamma=0.01,
                 delta=5.,epsilon=0.005,zeta=10.,fidelity_weight=0.01):
        super().__init__()
        self.alpha=alpha; self.beta=beta; self.gamma=gamma
        self.delta=delta; self.epsilon=epsilon; self.zeta=zeta
        self.fidelity_loss=QuantumFidelityLoss(n_qubits,fidelity_weight)
    def forward(self,quantum_output,y_true=None):
        var=quantum_output.var(0).mean()
        mmag=quantum_output.abs().mean()
        qstd=quantum_output.std(-1).mean()
        L1=self.alpha*torch.exp(-self.beta*var)
        L2=self.gamma*torch.exp(-self.delta*mmag)
        L3=self.epsilon*torch.exp(-self.zeta*qstd)
        L4=self.fidelity_loss(quantum_output,y_true) if y_true is not None else torch.tensor(0.,device=quantum_output.device)
        return L1+L2+L3+L4
    def breakdown(self,quantum_output,y_true=None):
        var=quantum_output.var(0).mean(); mmag=quantum_output.abs().mean(); qstd=quantum_output.std(-1).mean()
        return {'barren':(self.alpha*torch.exp(-self.beta*var)).item(),
                'concentration':(self.gamma*torch.exp(-self.delta*mmag)).item(),
                'entanglement':(self.epsilon*torch.exp(-self.zeta*qstd)).item(),
                'fidelity':self.fidelity_loss(quantum_output,y_true).item() if y_true is not None else 0.}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QISKIT VQC BUILDERS  (from vqc_builder.py)
# ══════════════════════════════════════════════════════════════════════════════

ENCODING_OPTIONS=['angle','angle_full','dense','iqp','pauli','arctan','fourier','amplitude']
ANSATZ_OPTIONS=['u_ring','u_full','u_alternate','efficient_su2','real_amplitudes','strongly','hardware','excitation']

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import QFTGate, efficient_su2, real_amplitudes as _ra, excitation_preserving, n_local
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    _QISKIT_OK = True
except ImportError:
    _QISKIT_OK = False

def _req_qiskit():
    if not _QISKIT_OK: raise ImportError("Qiskit not installed. pip install qiskit qiskit-machine-learning")

def build_vqc(n_qubits,n_layers,encoding='angle_full',ansatz='efficient_su2',reuploading=True,init_scale=0.1):
    """
    Build a Qiskit QuantumCircuit with data encoding + trainable ansatz.
    Returns (qc, x_params, w_params).

    Encodings : angle | angle_full | dense | iqp | pauli | arctan | fourier | amplitude
    Ansatz    : u_ring | u_full | u_alternate | efficient_su2 | real_amplitudes |
                strongly | hardware | excitation
    """
    _req_qiskit()
    nq=n_qubits; nl=n_layers
    x_params=ParameterVector('x',nq)

    # Encoding functions
    def enc(qc):
        if encoding=='angle':
            for q in range(nq): qc.ry(x_params[q],q)
        elif encoding=='angle_full':
            for q in range(nq): qc.rx(x_params[q],q); qc.ry(x_params[q],q); qc.rz(x_params[q],q)
        elif encoding=='dense':
            for q in range(nq): qc.rx(x_params[q],q); qc.rz(x_params[q],q)
        elif encoding=='iqp':
            for q in range(nq): qc.h(q)
            for q in range(nq): qc.rz(x_params[q]*x_params[q],q)
            for q in range(nq-1): qc.rzz(x_params[q]*x_params[q+1],q,q+1)
        elif encoding=='pauli':
            for _ in range(2):
                for q in range(nq): qc.h(q)
                for q in range(nq): qc.rz(2*x_params[q],q)
                for q in range(nq-1):
                    qc.cx(q,q+1); qc.rz(2*(math.pi-x_params[q])*(math.pi-x_params[q+1]),q+1); qc.cx(q,q+1)
        elif encoding=='arctan':
            for q in range(nq): qc.ry(x_params[q]*(math.pi/4),q)
        elif encoding=='fourier':
            for q in range(nq): qc.ry(x_params[q],q)
            qc.append(QFTGate(nq),list(range(nq)))
        elif encoding=='amplitude':
            for q in range(nq): qc.h(q); qc.rz(x_params[q],q); qc.ry(x_params[q],q)
        else:
            raise ValueError(f"Unknown encoding {encoding!r}. Options: {ENCODING_OPTIONS}")

    # Ansatz functions
    def make_ans():
        if ansatz=='u_ring':
            w=ParameterVector('w',nl*nq*3); qc=QuantumCircuit(nq); wi=0
            for _ in range(nl):
                for q in range(nq): qc.u(w[wi],w[wi+1],w[wi+2],q); wi+=3
                for q in range(nq-1): qc.cx(q,q+1)
                qc.cx(nq-1,0)
            return qc,list(w)
        elif ansatz=='u_full':
            w=ParameterVector('w',nl*nq*3); qc=QuantumCircuit(nq); wi=0
            for _ in range(nl):
                for q in range(nq): qc.u(w[wi],w[wi+1],w[wi+2],q); wi+=3
                for q in range(nq):
                    for r in range(q+1,nq): qc.cx(q,r)
            return qc,list(w)
        elif ansatz=='u_alternate':
            w=ParameterVector('w',nl*nq*3); qc=QuantumCircuit(nq); wi=0
            for l in range(nl):
                for q in range(nq): qc.u(w[wi],w[wi+1],w[wi+2],q); wi+=3
                if l%2==0:
                    for q in range(nq-1): qc.cx(q,q+1)
                else:
                    for q in range(nq-1,0,-1): qc.cx(q,q-1)
            return qc,list(w)
        elif ansatz=='efficient_su2':
            a=efficient_su2(nq,reps=nl); return a,list(a.parameters)
        elif ansatz=='real_amplitudes':
            a=_ra(nq,reps=nl); return a,list(a.parameters)
        elif ansatz=='strongly':
            w=ParameterVector('w',nl*nq*3); qc=QuantumCircuit(nq); wi=0
            pairs=[(i,j) for i in range(nq) for j in range(i+1,nq)]
            for _ in range(nl):
                for q in range(nq): qc.rz(w[wi],q); wi+=1; qc.ry(w[wi],q); wi+=1; qc.rz(w[wi],q); wi+=1
                for qa,qb in pairs: qc.cz(qa,qb)
            return qc,list(w)
        elif ansatz=='hardware':
            a=n_local(nq,rotation_blocks=['ry','rz'],entanglement_blocks='cx',entanglement='linear',reps=nl)
            return a,list(a.parameters)
        elif ansatz=='excitation':
            a=excitation_preserving(nq,reps=nl); return a,list(a.parameters)
        else:
            raise ValueError(f"Unknown ansatz {ansatz!r}. Options: {ANSATZ_OPTIONS}")

    ans_qc, w_params = make_ans()
    qc = QuantumCircuit(nq)
    enc(qc)
    qc.compose(ans_qc, inplace=True)
    if reuploading:
        for _ in range(nl-1):
            enc(qc)
            qc.compose(ans_qc, inplace=True)
    return qc, x_params, w_params


def build_torch_vqc(n_qubits,n_layers,n_outputs=NO,encoding='angle_full',
                    ansatz='efficient_su2',reuploading=True,init_scale=0.1,
                    ql_alpha=0.01,ql_gamma=0.01,ql_epsilon=0.005,ql_fidelity_weight=0.01,
                    pde_mode='separate',pde_w_phys=0.1,pde_w_fid=0.01,pde_w_recon=0.01,
                    pde_init_nu=1.8e-5,pde_init_kappa=0.026):
    """
    Build a full Qiskit-backed VQC as a differentiable PyTorch module.

    Returns (torch_qnn, readout, qc, quantum_loss, pde_loss)

    Note: backward through TorchConnector can have shape issues;
    use ScalableQPINN (native PyTorch) for training, this for circuit analysis.
    """
    _req_qiskit()
    qc,x_params,w_params=build_vqc(n_qubits,n_layers,encoding,ansatz,reuploading,init_scale)
    n_obs=n_qubits
    observables=[SparsePauliOp('I'*(n_qubits-k-1)+'Z'+'I'*k) for k in range(n_obs)]
    qnn=EstimatorQNN(circuit=qc,estimator=StatevectorEstimator(),
                     observables=observables,input_params=list(x_params),
                     weight_params=w_params,input_gradients=True)
    init_w=torch.randn(len(w_params))*init_scale
    torch_qnn=TorchConnector(qnn,initial_weights=init_w)
    readout=nn.Linear(n_obs,n_outputs) if n_obs<n_outputs else nn.Identity()
    quantum_loss=QuantumLoss(n_qubits,ql_alpha,0.01,ql_gamma,5.,ql_epsilon,10.,ql_fidelity_weight)
    pde_loss=NSQuantumPDELoss(n_qubits,pde_mode,pde_w_phys,pde_w_fid,pde_w_recon,pde_init_nu,pde_init_kappa)
    return torch_qnn,readout,qc,quantum_loss,pde_loss


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — NATIVE PYTORCH STATEVECTOR VQC  (from ns_models.py)
# ══════════════════════════════════════════════════════════════════════════════

def _rx(t):
    c=torch.cos(t/2); s=torch.sin(t/2); z=torch.zeros_like(c)
    return torch.stack([torch.stack([c,-1j*s],1),torch.stack([-1j*s,c],1)],1).cfloat()
def _ry(t):
    c=torch.cos(t/2); s=torch.sin(t/2)
    return torch.stack([torch.stack([c,-s],1),torch.stack([s,c],1)],1).cfloat()
def _rz(t):
    ep=torch.exp(-1j*t/2); em=torch.exp(1j*t/2); z=torch.zeros_like(ep)
    return torch.stack([torch.stack([ep,z],1),torch.stack([z,em],1)],1).cfloat()
def _h_gate(B,dev):
    h=torch.tensor([[1,1],[1,-1]],dtype=torch.cfloat,device=dev)/math.sqrt(2)
    return h.unsqueeze(0).expand(B,-1,-1)
def _apply_1q(psi,G,q,n_q):
    B=psi.shape[0]; lo=2**q; hi=2**(n_q-q-1)
    psi_r=psi.reshape(B,lo,2,hi)
    if G.dim()==2: out=torch.einsum('ab,BjbH->BjaH',G.cfloat(),psi_r)
    else:          out=torch.einsum('Bab,BjbH->BjaH',G.cfloat(),psi_r)
    return out.reshape(B,2**n_q)
def _crz(theta,ctrl,tgt,psi,n_q):
    D=2**n_q; out=psi.clone()
    for i in range(D):
        if (i>>ctrl)&1:
            phase=torch.exp(-1j*theta/2) if not((i>>tgt)&1) else torch.exp(1j*theta/2)
            out[:,i]=psi[:,i]*phase
    return out
def _cry(theta,ctrl,tgt,psi,n_q):
    D=2**n_q; out=psi.clone(); c_=torch.cos(theta/2); s_=torch.sin(theta/2)
    for i in range(D):
        if (i>>ctrl)&1:
            j=i^(1<<tgt); bit_t=(i>>tgt)&1
            if bit_t==0 and j>i:
                out[:,i]=c_*psi[:,i]-s_*psi[:,j]; out[:,j]=s_*psi[:,i]+c_*psi[:,j]
    return out
def _zz(theta,q0,q1,psi,n_q):
    D=2**n_q; out=psi.clone()
    for i in range(D):
        b0=(i>>q0)&1; b1=(i>>q1)&1
        phase=torch.exp(1j*theta/2) if b0^b1 else torch.exp(-1j*theta/2)
        out[:,i]=psi[:,i]*phase
    return out
def _swap(psi,q0,q1,n_q):
    D=2**n_q; out=psi.clone()
    for i in range(D):
        b0=(i>>q0)&1; b1=(i>>q1)&1
        if b0!=b1:
            j=(i^(1<<q0))^(1<<q1)
            if j>i: out[:,i],out[:,j]=psi[:,j].clone(),psi[:,i].clone()
    return out
def _expval_z(psi,q,n_q):
    D=2**n_q; probs=psi.abs()**2
    signs=torch.tensor([-1. if (i>>q)&1 else 1. for i in range(D)],dtype=torch.float32,device=psi.device)
    return (probs*signs).sum(1)
def _expval_x(psi,q,n_q):
    B=psi.shape[0]; H=_h_gate(B,psi.device).squeeze(0)
    return _expval_z(_apply_1q(psi,H,q,n_q),q,n_q)
def _expval_y(psi,q,n_q):
    sdg=torch.tensor(-math.pi/2,device=psi.device).unsqueeze(0); B=psi.shape[0]
    psi2=_apply_1q(psi,_rz(sdg).squeeze(0),q,n_q)
    return _expval_z(_apply_1q(psi2,_h_gate(B,psi.device).squeeze(0),q,n_q),q,n_q)
def _expval_zz(psi,q0,q1,n_q):
    D=2**n_q; probs=psi.abs()**2
    signs=torch.tensor([1. if ((i>>q0)&1)==((i>>q1)&1) else -1. for i in range(D)],dtype=torch.float32,device=psi.device)
    return (probs*signs).sum(1)


def _rich_vqc(x,params,n_q,n_l,n_q_total=None,sys_offset=0):
    """Rich-gate VQC: RX+RY+RZ data encoding, RX+RY+RZ trainable, CRZ+CRY+ZZ all-to-all."""
    if n_q_total is None: n_q_total=n_q
    B=x.shape[0]; dev=x.device; n_enc=min(x.shape[1],n_q)
    sys_q=[sys_offset+i for i in range(n_q)]
    pairs=[(sys_q[i],sys_q[j]) for i in range(n_q) for j in range(i+1,n_q)]
    psi=torch.zeros(B,2**n_q_total,dtype=torch.cfloat,device=dev); psi[:,0]=1.
    H=_h_gate(B,dev)
    for q in range(n_q_total): psi=_apply_1q(psi,H,q,n_q_total)
    for l in range(n_l):
        for qi,q in enumerate(sys_q[:n_enc]):
            psi=_apply_1q(psi,_rx(x[:,qi]),q,n_q_total)
            psi=_apply_1q(psi,_ry(x[:,qi]),q,n_q_total)
            psi=_apply_1q(psi,_rz(x[:,qi]**2),q,n_q_total)
        for qi,q in enumerate(sys_q):
            psi=_apply_1q(psi,_rx(params['w_rx'][l,qi].expand(B)),q,n_q_total)
            psi=_apply_1q(psi,_ry(params['w_ry'][l,qi].expand(B)),q,n_q_total)
            psi=_apply_1q(psi,_rz(params['w_rz'][l,qi].expand(B)),q,n_q_total)
        for pi,(qa,qb) in enumerate(pairs):
            psi=_crz(params['w_crz'][l,pi],qa,qb,psi,n_q_total)
            psi=_cry(params['w_cry'][l,pi],qa,qb,psi,n_q_total)
            psi=_zz( params['w_zz'][l,pi], qa,qb,psi,n_q_total)
        if l%2==1: psi=_swap(psi,sys_q[0],sys_q[-1],n_q_total)
    return psi, sys_q


def _validate_nq(n,model_type='qpinn'):
    if not isinstance(n,int) or n<NQ_MIN:
        raise ValueError(f"n_qubits must be int ≥ {NQ_MIN}, got {n!r}")
    if n>NQ_MAX:
        raise ValueError(f"n_qubits={n} > {NQ_MAX}: statevector 2^{n} too slow on CPU. Use GPU.")
    if model_type=='qpinn' and n<NI:
        miss=['x','t','p_ratio','mu','rho_L','rho_R','p_R'][n:]
        warnings.warn(f"ScalableQPINN nq={n} encodes only {n}/{NI} features. Missing: {miss}. "
                      f"Use nq≥{NI} or 'qapinn_scalable' (hybrid).",UserWarning,stacklevel=3)
    return n


QPINN_PRESETS={
    'nano':   (2,4,'dim=4, R²≈0.05, ~0.5s/ep'),
    'shallow':(3,4,'dim=8, R²≈0.27, ~1s/ep'),
    'medium': (4,6,'dim=16,R²≈0.50, ~4s/ep'),
    'deep':   (5,8,'dim=32,R²≈0.70, ~13s/ep'),
    'wide':   (6,6,'dim=64,R²≈0.88, ~30s/ep'),
    'full':   (6,10,'dim=64,R²≈0.90, ~46s/ep'),
    'large':  (7,8,'dim=128,R²≈0.97, ~85s/ep'),
    'max':    (8,6,'dim=256,R²≈0.99, ~250s/ep'),
}


class ScalableQPINN(nn.Module):
    """
    Full-quantum PINN with rich gate set and selectable depth/width.
    Gates: RX+RY+RZ data-reupload + RX+RY+RZ trainable + CRZ+CRY+ZZ all-to-all + H init + SWAP.
    Input: first n_qubits features of NI=7 are encoded (nq≥7 for full coverage).
    """
    def __init__(self,n_qubits=8,n_layers=4,preset=None,use_physics=False):
        super().__init__()
        if preset:
            n_qubits,n_layers,_=QPINN_PRESETS[preset]
        self.n_q=_validate_nq(n_qubits,'qpinn'); self.n_l=n_layers
        self.use_physics=use_physics; self.n_pairs=self.n_q*(self.n_q-1)//2
        self.w_rx =nn.Parameter(torch.randn(n_layers,self.n_q)*0.1)
        self.w_ry =nn.Parameter(torch.randn(n_layers,self.n_q)*0.1)
        self.w_rz =nn.Parameter(torch.randn(n_layers,self.n_q)*0.1)
        self.w_crz=nn.Parameter(torch.randn(n_layers,self.n_pairs)*0.1)
        self.w_cry=nn.Parameter(torch.randn(n_layers,self.n_pairs)*0.1)
        self.w_zz =nn.Parameter(torch.randn(n_layers,self.n_pairs)*0.1)
        self.scale=nn.Parameter(torch.ones(NO)*0.5)
        self.shift=nn.Parameter(torch.zeros(NO))
    def forward(self,x):
        p={'w_rx':self.w_rx,'w_ry':self.w_ry,'w_rz':self.w_rz,
           'w_crz':self.w_crz,'w_cry':self.w_cry,'w_zz':self.w_zz}
        psi,_=_rich_vqc(x[:,:self.n_q],p,self.n_q,self.n_l)
        out=torch.stack([_expval_z(psi,q,self.n_q) for q in range(NO)],1)
        return out*self.scale+self.shift
    @property
    def hilbert_dim(self): return 2**self.n_q


class ScalableQAPINN(nn.Module):
    """Hybrid QAPINN: classical encoder (7→nq) → rich VQC → classical decoder (nq→3)."""
    def __init__(self,n_qubits=8,n_layers=4,hidden=128,preset=None,use_physics=False):
        super().__init__()
        if preset: n_qubits,n_layers,_=QPINN_PRESETS[preset]
        self.n_q=_validate_nq(n_qubits,'qapinn'); self.n_l=n_layers
        self.use_physics=use_physics; self.n_pairs=self.n_q*(self.n_q-1)//2
        self.encoder=nn.Sequential(nn.Linear(NI,hidden),nn.Tanh(),nn.Linear(hidden,self.n_q),nn.Tanh())
        self.w_rx =nn.Parameter(torch.randn(n_layers,self.n_q)*0.1)
        self.w_ry =nn.Parameter(torch.randn(n_layers,self.n_q)*0.1)
        self.w_rz =nn.Parameter(torch.randn(n_layers,self.n_q)*0.1)
        self.w_crz=nn.Parameter(torch.randn(n_layers,self.n_pairs)*0.1)
        self.w_cry=nn.Parameter(torch.randn(n_layers,self.n_pairs)*0.1)
        self.w_zz =nn.Parameter(torch.randn(n_layers,self.n_pairs)*0.1)
        self.decoder=nn.Sequential(nn.Tanh(),nn.Linear(NO,hidden),nn.Tanh(),nn.Linear(hidden,NO))
    def forward(self,x):
        z=self.encoder(x)
        p={'w_rx':self.w_rx,'w_ry':self.w_ry,'w_rz':self.w_rz,
           'w_crz':self.w_crz,'w_cry':self.w_cry,'w_zz':self.w_zz}
        psi,_=_rich_vqc(z,p,self.n_q,self.n_l)
        out=torch.stack([_expval_z(psi,q,self.n_q) for q in range(NO)],1)
        return self.decoder(out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUANTUM FORWARD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def forward_with_raw(model: ScalableQPINN, x: torch.Tensor):
    """
    Run ScalableQPINN and return both prediction AND raw ⟨Z⟩ values.

    Required for LCU loss: NSQuantumPDELoss expects raw_Z (B, n_qubits)
    while the model's forward() only returns pred (B, 3).

    Parameters
    ----------
    model : ScalableQPINN
    x     : (B, NI) input tensor

    Returns
    -------
    pred  : (B, 3)       — [ρ̂, û, p̂]
    raw_Z : (B, n_qubits) — ⟨Z_q⟩ for all qubits
    """
    p={'w_rx':model.w_rx,'w_ry':model.w_ry,'w_rz':model.w_rz,
       'w_crz':model.w_crz,'w_cry':model.w_cry,'w_zz':model.w_zz}
    psi,_=_rich_vqc(x[:,:model.n_q],p,model.n_q,model.n_l)
    raw_Z=torch.stack([_expval_z(psi,q,model.n_q) for q in range(model.n_q)],1)
    pred=raw_Z[:,:NO]*model.scale+model.shift
    return pred, raw_Z


def get_device(prefer_gpu=True):
    """Select best available device: CUDA → MPS → CPU."""
    if not prefer_gpu: return torch.device('cpu')
    if torch.cuda.is_available():
        name=torch.cuda.get_device_name(0); mem=torch.cuda.get_device_properties(0).total_memory/1e9
        print(f"[Device] CUDA — {name} ({mem:.1f} GB VRAM)"); return torch.device('cuda')
    if torch.backends.mps.is_available():
        print("[Device] MPS — Apple Silicon (quantum ops fall back to CPU)"); return torch.device('mps')
    print("[Device] CPU"); return torch.device('cpu')


def model_to_device(model,device):
    """Move model to device; keeps quantum models on CPU if device=MPS."""
    if device.type=='mps' and isinstance(model,(ScalableQPINN,ScalableQAPINN,NSQuantumPDELoss)):
        warnings.warn("MPS doesn't support complex64. Quantum model stays on CPU.",UserWarning)
        return model
    return model.to(device)


def estimate_vram(model,batch_size=512):
    """Estimate peak VRAM (GB) for a model+batch_size."""
    n_params=sum(p.numel() for p in model.parameters())
    model_gb=n_params*4*3/1e9   # params + grads + Adam
    is_q=isinstance(model,(ScalableQPINN,ScalableQAPINN))
    if not is_q:
        return {'model_gb':round(model_gb,3),'forward_gb':round(batch_size*128*4*4*2/1e9,3),
                'tape_gb':0.,'total_gb':round(model_gb+batch_size*128*4*4*2/1e9,3),
                'fits_8gb':True,'rec_batch_8gb':4096}
    nq=getattr(model,'n_q',3); n_tot=getattr(model,'n_tot',nq)
    nl=getattr(model,'n_l',4); n_pairs=nq*(nq-1)//2; n_gates=(nq*6+n_pairs*3)*nl
    bps=2**n_tot*8*(1+n_gates)/1e9
    total_gb=model_gb+bps*batch_size
    budget=6.5; rec_b=max(16,min(4096,2**int(math.log2(max((budget-model_gb)/bps,1)))))
    return {'model_gb':round(model_gb,3),'forward_gb':round(2**n_tot*8*batch_size/1e9,3),
            'tape_gb':round(2**n_tot*8*n_gates*batch_size/1e9,3),
            'total_gb':round(total_gb,3),'fits_8gb':total_gb<6.5,
            'fits_16gb':total_gb<14.,'rec_batch_8gb':rec_b}


def vram_table(batch_size=512):
    """Print VRAM estimates for all standard QPINN configs."""
    configs=[('ScalableQPINN',nq,4) for nq in range(3,9)]
    print(f"\nVRAM at batch_size={batch_size}  (★ = default nq=8)")
    print(f"{'Model':<30} {'Params':>7} {'Total GB':>10} {'8GB':>6} {'RecB@8GB':>10}")
    print("─"*67)
    for _,nq,nl in configs:
        m=ScalableQPINN(n_qubits=nq,n_layers=nl)
        info=estimate_vram(m,batch_size); n=sum(p.numel() for p in m.parameters())
        ok='OK' if info['fits_8gb'] else 'OOM'
        star=' ★ DEFAULT' if nq==8 else ''
        print(f"  ScalableQPINN nq={nq:<2}           {n:>7,} {info['total_gb']:>10.3f} {ok:>6} {info['rec_batch_8gb']:>10}{star}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — QUBIT / PRESET UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def qubit_info(n_qubits: int) -> str:
    """Human-readable summary: feature coverage, Hilbert dim, R² estimate, epoch time."""
    _validate_nq(n_qubits,'info')
    nq=n_qubits; dim=2**nq; n_pairs=nq*(nq-1)//2
    est_r2={2:0.05,3:0.18,4:0.38,5:0.59,6:0.82,7:0.97,8:0.99}.get(nq,'?')
    est_t ={2:0.5,3:1.2,4:4,5:13,6:30,7:85,8:250}.get(nq,'?')
    FEAT=['x','t','p_ratio','mu','rho_L','rho_R','p_R']
    n_enc=min(nq,len(FEAT)); enc=FEAT[:n_enc]; miss=FEAT[n_enc:]
    feat_note=f"ALL {len(FEAT)} features {enc} ✓" if not miss else f"{n_enc}/{len(FEAT)} encoded {enc}, MISSING {miss}"
    return (f"n_qubits={nq}:\n  Feature encoding : {feat_note}\n"
            f"  Hilbert space    : 2^{nq} = {dim}\n  Pairs           : {n_pairs}\n"
            f"  Params @ nl=6    : {nq*3*6+n_pairs*3*6+6}\n"
            f"  Est R2 max       : {est_r2}\n  Est epoch time   : {est_t}s\n")



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — IBM QUANTUM HARDWARE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def ibm_backend(token: str, n_qubits: int = 8, instance: str = 'ibm-q/open/main',
                least_busy: bool = True, simulator: bool = False):
    """
    Connect to IBM Quantum and return a real backend for nq-qubit circuits.

    Requirements
    ------------
        pip install qiskit-ibm-runtime

    Parameters
    ----------
    token       : str  — IBM Quantum API token (from https://quantum.ibm.com/)
    n_qubits    : int  — minimum qubit count needed (default 8)
    instance    : str  — IBM Quantum instance (default open/free plan)
    least_busy  : bool — pick the least-busy backend (fastest queue)
    simulator   : bool — use ibm_qasm_simulator instead of real hardware

    Returns
    -------
    backend : IBMBackend  — pass to transpile() and run circuits on

    Example
    -------
    >>> backend = ibm_backend(token='YOUR_TOKEN', n_qubits=8)
    >>> qc, x_params, w_params = build_vqc(8, n_layers=2)
    >>> from qiskit import transpile
    >>> qc_t = transpile(qc, backend=backend, optimization_level=3)
    >>> print(backend.name, backend.num_qubits, backend.status().pending_jobs)
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime not installed.\n"
            "  pip install qiskit-ibm-runtime")

    service = QiskitRuntimeService(channel='ibm_quantum', token=token, instance=instance)

    if simulator:
        backend = service.backend('ibm_qasm_simulator')
        print(f"[IBM] Using simulator: {backend.name}")
        return backend

    backends = service.backends(
        filters=lambda b: (b.num_qubits >= n_qubits
                           and b.status().operational
                           and not b.configuration().simulator),
        operational=True,
    )
    if not backends:
        raise RuntimeError(
            f"No operational IBM backend with ≥{n_qubits} qubits found.\n"
            f"  Available backends: {[b.name for b in service.backends()]}")

    if least_busy:
        backend = min(backends, key=lambda b: b.status().pending_jobs)
    else:
        backend = backends[0]

    cfg = backend.configuration()
    sts = backend.status()
    print(f"[IBM] Backend      : {backend.name}")
    print(f"      Qubits       : {cfg.num_qubits}")
    print(f"      Basis gates  : {cfg.basis_gates}")
    print(f"      Pending jobs : {sts.pending_jobs}")
    print(f"      Queue status : {'operational' if sts.operational else 'NOT operational'}")
    return backend


def run_on_ibm(qc, x_vals: torch.Tensor, w_vals: torch.Tensor,
               backend, x_params, w_params, shots: int = 1024):
    """
    Run a parameterised Qiskit circuit on an IBM backend.

    Parameters
    ----------
    qc       : QuantumCircuit        — from build_vqc()
    x_vals   : (B, n_qubits) tensor  — input values (one circuit per sample)
    w_vals   : (n_params,) tensor    — trained VQC weights
    backend  : IBMBackend            — from ibm_backend()
    x_params : ParameterVector       — x parameters from build_vqc()
    w_params : list                  — weight parameters from build_vqc()
    shots    : int                   — measurement shots per circuit

    Returns
    -------
    results : (B, n_qubits) float32 tensor — ⟨Z_q⟩ expectation values

    Notes
    -----
    Real hardware execution is SLOW (minutes per batch due to queue).
    Use for validation on hardware, not for training.
    Training should always use ScalableQPINN (CPU/GPU statevector).
    """
    _req_qiskit()
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler, Session
        from qiskit import transpile
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        raise ImportError("pip install qiskit-ibm-runtime")

    B = x_vals.shape[0]; nq = qc.num_qubits
    w_np = w_vals.detach().cpu().numpy()

    # Transpile once
    qc_t = transpile(qc, backend=backend, optimization_level=3)

    all_exp = []
    with Session(backend=backend) as session:
        sampler = Sampler(session=session)
        for b in range(B):
            x_np = x_vals[b].detach().cpu().numpy()
            param_vals = {x_params[i]: float(x_np[i]) for i in range(len(x_params))}
            param_vals.update({w_params[i]: float(w_np[i]) for i in range(len(w_params))})
            bound = qc_t.assign_parameters(param_vals)
            bound.measure_all()
            job = sampler.run([bound], shots=shots)
            counts = job.result()[0].data.meas.get_counts()
            # Compute ⟨Z_q⟩ = (n_0 - n_1) / shots for each qubit
            exp_z = []
            for q in range(nq):
                n0 = sum(v for k,v in counts.items() if k[-(q+1)] == '0')
                n1 = sum(v for k,v in counts.items() if k[-(q+1)] == '1')
                exp_z.append((n0 - n1) / shots)
            all_exp.append(exp_z)

    return torch.tensor(all_exp, dtype=torch.float32)


def get_ibm_noise_model(backend):
    """
    Build a Qiskit Aer noise model from a real IBM backend's calibration data.
    Use this for noisy simulation (faster than real hardware, more realistic
    than statevector).

    Returns
    -------
    noise_model : NoiseModel
    coupling_map, basis_gates for transpile()

    Example
    -------
    >>> noise_model, coupling_map, basis_gates = get_ibm_noise_model(backend)
    >>> from qiskit_aer import AerSimulator
    >>> sim = AerSimulator(noise_model=noise_model)
    """
    try:
        from qiskit_aer.noise import NoiseModel
    except ImportError:
        raise ImportError("pip install qiskit-aer")
    noise_model   = NoiseModel.from_backend(backend)
    coupling_map  = backend.configuration().coupling_map
    basis_gates   = noise_model.basis_gates
    print(f"[Noise] Model from {backend.name}")
    print(f"  Basis gates  : {basis_gates}")
    print(f"  Coupling map : {len(coupling_map)} connections")
    return noise_model, coupling_map, basis_gates


def list_ibm_backends(token: str, instance: str = 'ibm-q/open/main', min_qubits: int = 1):
    """
    List all available IBM Quantum backends with qubit count and queue depth.

    Example
    -------
    >>> list_ibm_backends(token='YOUR_TOKEN')
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError("pip install qiskit-ibm-runtime")
    service = QiskitRuntimeService(channel='ibm_quantum', token=token, instance=instance)
    print(f"\n{'Backend':<30} {'Qubits':>8} {'Pending':>10} {'Status'}")
    print("─"*60)
    for b in sorted(service.backends(), key=lambda x: x.num_qubits):
        if b.num_qubits < min_qubits: continue
        try:
            sts = b.status()
            op  = 'OK' if sts.operational else 'DOWN'
            print(f"  {b.name:<28} {b.num_qubits:>8} {sts.pending_jobs:>10}  {op}")
        except Exception:
            print(f"  {b.name:<28} {b.num_qubits:>8}  (status unavailable)")


if __name__ == '__main__':
    print("utilities_quantum.py — self test")
    print("─"*50)
    nq=4; B=4
    print(f"\n[1] NSQuantumPDELoss nq={nq}")
    pde=NSQuantumPDELoss(n_qubits=nq)
    raw=torch.tanh(torch.randn(B,nq)); y=torch.randn(B,3)
    loss=pde(raw,y_true=y)
    loss.backward()
    print(f"  loss={loss.item():.4f}  grad OK={pde.log_nu.grad is not None}")
    bd=pde.breakdown(raw,y)
    for eq in ['mass','momentum','energy','fidelity']:
        print(f"  {eq}: {bd[eq]}")

    print(f"\n[2] QuantumLoss nq={nq}")
    ql=QuantumLoss(nq)
    print(f"  loss={ql(raw,y).item():.4f}  breakdown={ql.breakdown(raw,y)}")

    print(f"\n[3] ScalableQPINN nq={nq}")
    m=ScalableQPINN(n_qubits=nq,n_layers=3)
    m._Ym=torch.zeros(3); m._Ys=torch.ones(3)
    x7=torch.rand(B,7)
    pred,raw_Z=forward_with_raw(m,x7)
    print(f"  pred={pred.shape}  raw_Z={raw_Z.shape}")
    lcu_loss=pde(raw_Z,y_true=y)
    (nn.functional.mse_loss(pred,y)+0.1*lcu_loss).backward()
    print(f"  lcu_loss={lcu_loss.item():.4f}  grad OK")

    print(f"\n[4] qubit_info(7)")
    print(qubit_info(7))

    print("\n[5] vram_table(batch_size=256)")
    vram_table(256)
    print("Done ✓")
