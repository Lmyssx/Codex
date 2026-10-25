import numpy as np
C=299792458.

def wave_parameters(f_hz,eps_layers):
    f=np.asarray(f_hz,dtype=float);eps=np.asarray(eps_layers,dtype=complex)
    if f.ndim!=1 or eps.ndim!=2 or eps.shape[1]!=len(f) or not np.all(np.isfinite(eps)) or np.any(f<=0):
        raise ValueError('invalid layer/frequency array')
    if np.any(eps.real<=0) or np.any(eps.imag>1e-12): raise ValueError('positive passive dielectric required')
    n=np.sqrt(eps)
    if np.any(n.imag>1e-12): raise ValueError('nonattenuating square root branch')
    return 2*np.pi*f/C*n,376.730313668/n

def recursive_reflection(f_hz,eps_layers,finite_thickness_m,ambient_gap_m=0.,return_debug=False):
    eps=np.asarray(eps_layers);d=np.asarray(finite_thickness_m,float)
    if eps.shape[0]<2 or d.shape!=(eps.shape[0]-2,) or np.any(d<0) or not np.all(np.isfinite(d)) or ambient_gap_m<0:
        raise ValueError('invalid finite layer thicknesses')
    k,eta=wave_parameters(f_hz,eps)
    r=(eta[1:]-eta[:-1])/(eta[1:]+eta[:-1]);R=r[-1].copy();steps=[];factors=[]
    for i in range(len(d)-1,-1,-1):
        p=np.exp(-2j*k[i+1]*d[i]);q=R*p;den=1+r[i]*q
        if np.any(abs(den)<1e-14): raise FloatingPointError('near singular recursive denominator')
        R=(r[i]+q)/den;steps.append(R.copy());factors.append(p)
    R*=np.exp(-2j*k[0]*ambient_gap_m)
    if not np.all(np.isfinite(R)): raise FloatingPointError('nonfinite reflection')
    return {'reflection':R,'local_r':r,'propagation':factors,'steps':steps} if return_debug else R

def tissue_reflection(f,eps,d_skin_m,d_fat_m,g_air_m=0.):
    return recursive_reflection(f,np.array([eps[k] for k in ['air','skin','fat','muscle']]),[d_skin_m,d_fat_m],g_air_m)

def consistent_tmm(f,eps,d,g=0.):
    """Independent right-to-left field mapping; reference for moderate attenuation."""
    k,eta=wave_parameters(f,eps);r=(eta[1:]-eta[:-1])/(eta[1:]+eta[:-1]);result=[]
    for j in range(len(f)):
        p0=np.exp(-1j*k[0,j]*g);M=np.diag([1/p0,p0])
        for i in range(len(r)):
            M=M@(np.array([[1,r[i,j]],[r[i,j],1]])/(1+r[i,j]))
            if i<len(d):
                p=np.exp(-1j*k[i+1,j]*d[i]);M=M@np.diag([1/p,p])
        result.append(M[1,0]/M[0,0])
    return np.array(result)
