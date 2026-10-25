import numpy as np
from .types import GateOperator

def make_operator(fc_hz=7.29e9, bandwidth_hz=1.5e9, dt_s=2*0.0514/299792458,
                  nfft=96, gate_bins=tuple(range(1,8)), f_obs_hz=None):
    gate = np.asarray(gate_bins)
    if (gate.ndim != 1 or gate.size == 0 or not np.issubdtype(gate.dtype,np.integer)
        or len(np.unique(gate)) != len(gate) or np.any(gate<0) or np.any(gate>=nfft)):
        raise ValueError('invalid original-position gate bins')
    if not (dt_s>0 and bandwidth_hz>0 and int(nfft)==nfft):
        raise ValueError('invalid frequency metadata')
    fb = np.fft.fftshift(np.fft.fftfreq(nfft, dt_s))
    f = fc_hz + fb; mask = np.abs(fb)<=bandwidth_hz/2
    if np.any(f<=0): raise ValueError('RF grid must be positive')
    if f_obs_hz is not None and not np.allclose(f[mask],f_obs_hz,rtol=1e-14,atol=1e-6):
        raise ValueError('frequency_metadata_mismatch')
    A = np.exp(-2j*np.pi*fb[mask,None]*dt_s*gate[None,:])
    q,_ = np.linalg.qr(A,mode='reduced'); sv=np.linalg.svd(A,compute_uv=False)
    if np.linalg.matrix_rank(A)!=gate.size: raise ValueError('gate matrix rank deficient')
    return GateOperator(fc_hz,bandwidth_hz,dt_s,nfft,gate,f,mask,q,sv)

def apply_step5_operator(H_full, operator):
    H_full=np.asarray(H_full,dtype=np.complex128)
    if H_full.shape[-1]!=operator.nfft: raise ValueError('full grid required')
    h=np.fft.ifft(np.fft.ifftshift(H_full,axes=-1),axis=-1)
    hg=np.zeros_like(h); hg[...,operator.gate_bins]=h[...,operator.gate_bins]
    return np.fft.fftshift(np.fft.fft(hg,axis=-1),axes=-1)[...,operator.mask]

def build_projection(operator):
    return operator.Q.copy(),operator.singular_values.copy()

def project_observation(y,Q):
    z=np.asarray(y)@Q.conj()
    return np.concatenate((z.real,z.imag),axis=-1)

def subspace_error(y,Q):
    y=np.asarray(y);den=np.linalg.norm(y)
    return float(np.linalg.norm(y-y@Q.conj()@Q.T)/den) if den else 0.0
