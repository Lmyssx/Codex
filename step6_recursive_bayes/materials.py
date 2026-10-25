from pathlib import Path
import re
import numpy as np
from .types import MaterialModel

KEYS=['ef','del1','tau1','alf1','del2','tau2','alf2','sig','del3','tau3','alf3','del4','tau4','alf4']

def load_material_model(path):
    raw=Path(path).read_bytes()
    try: text=raw.decode('utf-8')
    except UnicodeDecodeError: text=raw.decode('cp1252')
    rows=[l.split('\t') for l in text.splitlines()]
    for i,cols in enumerate(rows):
        names=[x.split('(')[0].strip().lower() for x in cols]
        if set(KEYS)<=set(names): break
    else: raise ValueError('Cole-Cole header missing')
    ix={k:names.index(k) for k in KEYS};units={}
    for k,u in zip(['tau1','tau2','tau3','tau4'],['ps','ns','us','ms']):
        match=re.search(r'\((.*?)\)',cols[ix[k]])
        if not match: raise ValueError('missing relaxation time unit')
        unit=match.group(1);units[k]=unit
        if unit.replace('µ','u').replace('μ','u')!=u: raise ValueError('unrecognized time unit: '+unit)
    params={}
    for name in ['Skin','Fat','Muscle']:
        selected=[r for r in rows[i+1:] if len(r)>1 and r[1].strip()==name]
        if len(selected)!=1: raise ValueError('exact unique tissue required: '+name)
        p={k:float(selected[0][ix[k]]) for k in KEYS}
        if not all(np.isfinite(list(p.values()))): raise ValueError('missing/nonfinite required material parameter')
        params[name.lower()]=p
    return MaterialModel(params,str(path),units)

def cole_cole(f,p):
    f=np.asarray(f,dtype=float)
    if np.any(f<=0) or not np.all(np.isfinite(f)): raise ValueError('positive finite RF required')
    omega=2*np.pi*f;eps=np.full(f.shape,p['ef'],complex)+p['sig']/(1j*omega*8.8541878128e-12)
    for i,scale in enumerate([1e-12,1e-9,1e-6,1e-3],1):
        eps+=p['del'+str(i)]/(1+(1j*omega*p['tau'+str(i)]*scale)**(1-p['alf'+str(i)]))
    if np.any(eps.imag>1e-12): raise ValueError('nonpassive material')
    return eps

def evaluate_materials(f_hz,baseline,nuisance=(0.,0.)):
    if isinstance(baseline,MaterialModel): baseline={k:cole_cole(f_hz,p) for k,p in baseline.parameters.items()}
    out={k:v.copy() for k,v in baseline.items()}
    u,v=nuisance
    out['fat']=np.exp(u)*out['fat'].real+1j*np.exp(v)*out['fat'].imag
    out['air']=np.ones(len(f_hz),complex)
    return out
