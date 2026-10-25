import numpy as np
from scipy.special import gammaln

def log_theta_prior(theta,bounds,material_sds=(.1,.2),delay=False):
    t=np.asarray(theta);b=np.asarray(bounds)
    if t.shape!=(len(b),) or not np.all(np.isfinite(t)) or np.any(t<b[:,0]) or np.any(t>b[:,1]): return -np.inf
    lp=-np.log(b[0,1]-b[0,0])-np.log(b[1,1]-b[1,0])
    for i,sd in enumerate(material_sds,2):
        if i<len(t)-(1 if delay else 0): lp+=-.5*(t[i]/sd)**2-np.log(sd*np.sqrt(2*np.pi))
    if delay: lp+=-.5*t[-1]**2-.5*np.log(2*np.pi)
    return float(lp)

def log_s2_prior(s2,alpha,beta):
    if s2<=0 or not np.isfinite(s2): return -np.inf
    return alpha*np.log(beta)-gammaln(alpha)-(alpha+1)*np.log(s2)-beta/s2

def noise_conditional(alpha,beta,D,SSE,T=1.):
    if T<1 or SSE<0: raise ValueError('invalid Gibbs inputs')
    return alpha+D/(2*T), beta+SSE/(2*T)
