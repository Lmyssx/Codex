import json
from pathlib import Path
import numpy as np
from .types import Calibration

REQUIRED=['reference_plane','fast_time_origin','fft_iq_convention','background_scheme','hardware_settings_id',
          'reference_ids','repeat_count','method','independent_validation_error','independent_reference','offset_assumption']

def load_calibration(path,operator,expected_metadata=None):
    if path is None or not Path(path).exists(): raise FileNotFoundError('calibration_missing')
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if any(k not in meta for k in REQUIRED): raise ValueError('incomplete calibration provenance')
    if not meta['independent_reference'] or meta.get('synthetic',False): raise ValueError('independent measured calibration required for real inference')
    if not np.isfinite(meta['independent_validation_error']) or meta['independent_validation_error']<0 or meta['repeat_count']<1: raise ValueError('invalid independent validation')
    for k in ['reference_plane','fft_iq_convention','background_scheme','hardware_settings_id','method']:
        if not isinstance(meta[k],str) or not meta[k].strip():raise ValueError('empty calibration metadata: '+k)
    if expected_metadata is None: raise ValueError('target calibration metadata must be independently specified')
    for k in ['reference_plane','fast_time_origin','fft_iq_convention','background_scheme','hardware_settings_id']:
        if k not in expected_metadata or expected_metadata[k]!=meta[k]: raise ValueError('calibration metadata mismatch: '+k)
    with np.load(path,allow_pickle=False) as z:
        f=z['f_full_hz'];s=z['system_response_full'];l=z['residual_offset_full']
    if f.dtype!=np.float64 or f.shape!=(operator.nfft,) or not np.allclose(f,operator.f_full_hz,rtol=1e-14,atol=1e-6):
        raise ValueError('calibration full frequency grid mismatch')
    for a in [s,l]:
        if a.dtype!=np.complex128 or a.shape!=(1,operator.nfft) or not np.all(np.isfinite(a)): raise ValueError('invalid calibration channel/array')
    return Calibration(f,s,l,meta)

def fit_reference_response(Y_full,X_full,train_indices,validation_indices):
    """Reference-only least squares; caller must separately verify load transferability.

    Arrays [reference,repeat,frequency]. Keep at least one entire reference held out.
    """
    y=np.asarray(Y_full,complex);x=np.asarray(X_full,complex)
    train=np.asarray(train_indices);valid=np.asarray(validation_indices)
    if y.shape!=x.shape or y.ndim!=3 or len(valid)<1 or len(train)<2 or set(train)&set(valid):
        raise ValueError('disjoint reference-level training and validation required')
    L=[];S=[];conditions=[]
    for f in range(y.shape[-1]):
        design=np.column_stack([np.ones(x[train,:,f].size),x[train,:,f].ravel()])
        if np.linalg.matrix_rank(design)<2: raise ValueError('reference reflections not identifiable')
        coef=np.linalg.lstsq(design,y[train,:,f].ravel(),rcond=None)[0]
        L.append(coef[0]);S.append(coef[1]);conditions.append(np.linalg.cond(design))
    L=np.asarray(L);S=np.asarray(S)
    error=np.linalg.norm(y[valid]-L-S*x[valid])/max(np.linalg.norm(y[valid]),1e-300)
    return S[None],L[None],{'heldout_relative_error':float(error),'max_design_condition':float(max(conditions))}
