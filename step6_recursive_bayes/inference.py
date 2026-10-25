import numpy as np
from scipy.optimize import minimize
from .priors import log_s2_prior

class Posterior:
    def __init__(self,likelihood,logprior,bounds,alpha=3.,beta=2.):
        self.likelihood=likelihood;self.logprior=logprior;self.bounds=np.asarray(bounds,float)
        self.alpha=alpha;self.beta=beta
    def joint(self,theta,s2): return self.likelihood(theta,s2)+self.logprior(theta)+log_s2_prior(s2,self.alpha,self.beta)
    def profile_s2(self,theta):
        return (self.beta+self.likelihood.sse(theta)/2)/(self.alpha+1+self.likelihood.D/2)
    def marginal(self,theta):
        sse=self.likelihood.sse(theta)
        return self.logprior(theta)-(self.alpha+self.likelihood.D/2)*np.log(self.beta+sse/2)

def run_map(posterior,grid_shape=(19,80),max_starts=8):
    b=posterior.bounds
    seeds=[]
    for ds in np.linspace(*b[0],grid_shape[0]):
        for df in np.linspace(*b[1],grid_shape[1]):
            t=np.zeros(len(b));t[:2]=ds,df
            s=posterior.profile_s2(t);score=posterior.joint(t,s)
            seeds.append((score,t,s))
    seeds.sort(key=lambda r:r[0],reverse=True);starts=[]
    for score,t,s in seeds:
        if all(np.linalg.norm((t[:2]-u[:2])/[.2,3.])>1 for u in starts): starts.append(t)
        if len(starts)>=max_starts: break
    candidates=[]
    for t in starts:
        def obj(v):
            lp=posterior.logprior(v)
            if not np.isfinite(lp):return 1e100
            s=posterior.profile_s2(v);return -posterior.joint(v,s)
        fit=minimize(obj,t,method='Nelder-Mead',bounds=b,options={'maxiter':2500,'xatol':1e-8,'fatol':1e-8})
        s=posterior.profile_s2(fit.x)
        candidates.append({'theta':fit.x,'s2':s,'logposterior':posterior.joint(fit.x,s),
                           'loglike':posterior.likelihood(fit.x,s),'success':bool(fit.success),
                           'boundary':bool(np.any(np.isclose(fit.x,b[:,0],atol=1e-4)) or np.any(np.isclose(fit.x,b[:,1],atol=1e-4)))})
    candidates.sort(key=lambda r:r['logposterior'],reverse=True)
    return candidates,{'theta':np.array([r[1] for r in seeds]),'logposterior':np.array([r[0] for r in seeds])}

def dense_integral(posterior,axes):
    if len(posterior.bounds)!=2:raise ValueError('2D baseline integration only')
    a,b=axes;ll=np.array([[posterior.marginal([x,y]) for y in b] for x in a])
    # Trapezoidal quadrature includes half-weight at endpoints.
    wa=np.ones(len(a));wb=np.ones(len(b));wa[[0,-1]]=.5;wb[[0,-1]]=.5
    weight=np.exp(ll-ll.max())*wa[:,None]*wb[None,:];weight/=weight.sum()
    return weight

def map_mixture_proposal(posterior,candidates):
    """Frozen full-dimensional Student-t mixture; MH correction ensures exact target."""
    locations=[];covariances=[]
    for row in candidates:
        x=row['theta'];p=len(x);H=np.empty((p,p));h=1e-4
        if any(np.linalg.norm(x-v)<.01 for v in locations):continue
        if row['boundary']:continue
        for i in range(p):
            for j in range(p):
                a=np.eye(p)[i]*h;b=np.eye(p)[j]*h
                H[i,j]=-(posterior.marginal(x+a+b)-posterior.marginal(x+a-b)-posterior.marginal(x-a+b)+posterior.marginal(x-a-b))/(4*h*h)
        if not np.all(np.isfinite(H)):continue
        eig,V=np.linalg.eigh(H);C=(V/np.maximum(eig,1e-3))@V.T
        locations.append(x);covariances.append(C)
    if not locations:raise RuntimeError('no valid global mixture modes')
    weights=np.full(len(locations),.2/len(locations));weights[0]+=.8
    return {'means':np.array(locations).tolist(),'covariances':np.array(covariances).tolist(),'weights':weights.tolist()}
