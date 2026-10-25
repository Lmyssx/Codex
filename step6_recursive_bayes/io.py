import hashlib,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .types import ObservationRecord
from .operators import make_operator,subspace_error,apply_step5_operator,project_observation

def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_config(path):
    path=Path(path).resolve();c=json.loads(path.read_text(encoding='utf-8'))
    if c.get('schema_version')!=1: raise ValueError('unsupported schema_version')
    c['_project_root']=str(path.parent.parent)
    return c

def resolve(c,path):
    p=Path(path);return p if p.is_absolute() else Path(c['_project_root'])/p

def load_observation_record(path):
    path=Path(path)
    if not path.name.endswith('_observations.npz') or path.name.endswith('_diagnostics.npz'):
        raise ValueError('primary observations only')
    with np.load(path,allow_pickle=False) as z:
        required=['H_obs','f_obs','W','stride','fc','B_eff','Nfft','range_resolution',
                  'gate_bins','source_file','num_range_bins','num_frames','num_observations','num_frequency_bins','empty_subtraction']
        for k in required:
            if k not in z: raise ValueError('missing observation metadata: '+k)
        h=z['H_obs'];f=z['f_obs']
        if h.dtype!=np.complex128 or h.ndim!=3 or h.shape[1]!=1 or not np.all(np.isfinite(h)):
            raise ValueError('H_obs must be finite complex128[window,1,frequency]')
        if f.dtype!=np.float64 or f.ndim!=1 or np.any(f<=0) or not np.all(np.isfinite(f)) or np.any(np.diff(f)<=0):
            raise ValueError('invalid saved frequency axis')
        if h.shape[-1]!=len(f) or h.shape[0]!=int(z['num_observations']) or len(f)!=int(z['num_frequency_bins']):
            raise ValueError('array/metadata dimension mismatch')
        w=int(z['W']);s=int(z['stride']);n=int(z['num_range_bins']);nfft=int(z['Nfft'])
        if w<=0 or s<=0 or h.shape[0]!=(int(z['num_frames'])-w)//s+1 or n!=nfft:
            raise ValueError('inconsistent frame/range contract')
        op=make_operator(float(z['fc']),float(z['B_eff']),2*float(z['range_resolution'])/299792458,
                         nfft,z['gate_bins'],f)
        if subspace_error(h,op.Q)>1e-12: raise ValueError('observation_outside_gate_subspace')
        source=str(z['source_file'].item())
        meta={k:z[k].item() for k in ['empty_subtraction','num_frames']}
    meta.update({'processing_version':'step5_W110_original_gate','dt_source':'2*range_resolution/c from NPZ and build_observations.py',
                 'hardware_metadata_independently_verified':False,'subject_id':None,'site_id':None,'session_id':None})
    posture=path.parent.name.removesuffix('-observations')
    return ObservationRecord(h,f,op,w,s,n,source,posture,np.arange(h.shape[0]),file_hash(path),path,meta)

def scan_records(root):
    paths=sorted(Path(root).rglob('*_observations.npz'))
    if not paths: raise FileNotFoundError('observations_missing')
    return [load_observation_record(p) for p in paths if not p.name.endswith('_diagnostics.npz')]

def load_raw_mat(path):
    data=loadmat(path)
    if 'data' not in data: raise ValueError('MAT requires explicit data field')
    x=np.asarray(data['data'])
    if x.ndim!=2 or not np.iscomplexobj(x) or not np.all(np.isfinite(x)): raise ValueError('invalid complex CIR')
    return x.astype(np.complex128)

def raw_windows(x,record,background):
    if x.shape[1]!=record.range_bins: raise ValueError('raw range bins mismatch')
    return np.array([(x[i:i+record.W_frames]-background).mean(axis=0)
                     for i in range(0,len(x)-record.W_frames+1,record.stride_frames)])

def audit_records(records,raw_root=None,empty_subdir='2-307-空场'):
    empty_paths=sorted((Path(raw_root)/empty_subdir).glob('*.mat')) if raw_root else []
    background=np.mean([load_raw_mat(p).mean(axis=0) for p in empty_paths],axis=0) if empty_paths else None
    entries=[]
    for r in records:
        z=project_observation(r.H_obs[:,0],r.operator.Q)
        # Descriptive drift only: no thickness-based window screening.
        drift=float(np.linalg.norm(z-z.mean(axis=0))/max(np.linalg.norm(z),1e-300))
        entry={'path':str(r.path),'sha256':r.sha256,'source_file':r.source_file,'posture':r.posture,
               'shape':list(r.H_obs.shape),'rank_complex':int(len(r.operator.gate_bins)),
               'frequency_min_hz':float(r.f_obs_hz[0]),'frequency_max_hz':float(r.f_obs_hz[-1]),
               'frequency_step_hz':float(np.diff(r.f_obs_hz).mean()),'fc_hz':r.operator.fc_hz,
               'bandwidth_hz':r.operator.bandwidth_hz,'dt_s':r.operator.dt_s,'nfft':r.operator.nfft,
               'gate_bins':r.operator.gate_bins.tolist(),'W_frames':r.W_frames,'stride_frames':r.stride_frames,
               'subspace_error':subspace_error(r.H_obs,r.operator.Q),'relative_window_scatter':drift,
               'repeat_stability_review_required':bool(drift>.2),
               'raw_reconstruction':'raw_data_missing','coherence':'requires_original_frames',**r.metadata}
        raw=Path(raw_root)/('2-307-'+r.posture)/r.source_file if raw_root else None
        if raw and raw.exists() and background is not None:
            x=load_raw_mat(raw);hw=raw_windows(x,r,background)
            y=apply_step5_operator(np.fft.fftshift(np.fft.fft(hw,axis=-1),axes=-1),r.operator)[:,None]
            entry['raw_max_abs_error']=float(np.max(abs(y-r.H_obs)))
            if not np.allclose(y,r.H_obs,rtol=1e-12,atol=1e-14): raise ValueError('raw reconstruction mismatch')
            entry['raw_reconstruction']='passed'
            coherences=[]
            for start in range(0,len(x)-r.W_frames+1,r.stride_frames):
                gate=(x[start:start+r.W_frames]-background)[:,r.operator.gate_bins]
                denominator=np.mean(np.linalg.norm(gate,axis=-1))
                coherences.append(float(np.linalg.norm(gate.mean(axis=0))/denominator) if denominator else None)
            entry['coherence']=coherences
        if file_hash(r.path)!=r.sha256: raise RuntimeError('input modified')
        entries.append(entry)
    return {'status':'audit_only','record_count':len(records),'window_count':sum(len(r.H_obs) for r in records),
            'raw_empty_files':len(empty_paths),'entries':entries,'truth_available':False,
            'reference_calibration_available':False,'notes':['posture is not subject identity','range_resolution treated as sample spacing, hardware unverified',
            'README +/-1.50 GHz disagrees with code +/-0.75 GHz','raw coherence and empirical noise pending MAT upload']}
