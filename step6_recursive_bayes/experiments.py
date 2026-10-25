"""Adaptive importance integration for repeated conditional synthetic experiments."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from .likelihood import predict_many

def conditional_importance(obs,context,mode,rng,sd_factor=1.,sample_count=8192):
    """Known skin; free fat thickness and shared Gaussian nuisance. s2 integrated exactly.

    Mixture Student-t proposals fitted to multiple posterior modes. Returns explicit
    importance ESS and half-sample quantile disagreement; never assumes validity.
    """
    dim={'fixed':1,'material':3,'delay':2}[mode];D=obs.size
    sigma2=float(context.covariance[0,0]);alpha=context.alpha0+D/2
    def expand(v):
        v=np.atleast_2d(v);t=np.zeros((len(v),5 if mode=='delay' else 4));t[:,0]=1.5;t[:,1]=v[:,0]
        if mode=='material':t[:,2:4]=v[:,1:3]*np.array([.1,.2])*sd_factor
        if mode=='delay':t[:,4]=v[:,1]
        return t
    def logtarget(v):
        v=np.atleast_2d(v);valid=(v[:,0]>=1)&(v[:,0]<=60)&np.all(np.isfinite(v),axis=1)
        lp=np.full(len(v),-np.inf)
        if np.any(valid):
            t=v[valid];z=predict_many(expand(t),context)
            sse=np.sum((obs[None]-z[:,None])**2,axis=(1,2))/sigma2
            lp[valid]=-alpha*np.log(context.beta0+sse/2)-.5*np.sum(t[:,1:]**2,axis=1)
        return lp
    if mode=='fixed':
        grid=np.linspace(1.,60.,5901);logp=logtarget(grid[:,None]);w=np.exp(logp-logsumexp(logp))
        def q(x,weights):return np.interp([.025,.5,.975],np.cumsum(weights)-.5*weights,x)
        interval=q(grid,w);coarse=np.exp(logp[::2]-logsumexp(logp[::2]));difference=float(max(abs(interval-q(grid[::2],coarse))))
        return {'interval':interval,'importance_ess':None,'half_quantile_difference_mm':difference,
                'integration_adequate':difference<.015,'proposal_mode_locations':[],
                'sample_count':len(grid),'integration_method':'global dense 1D grid, 0.01 mm; coarsening check'}
    from scipy.signal import find_peaks
    grid=np.linspace(1,60,401);v=np.zeros((len(grid),dim));v[:,0]=grid;profile=logtarget(v)
    peaks,_=find_peaks(profile);peaks=sorted(peaks,key=lambda i:profile[i],reverse=True)[:12]
    starts=list(grid[peaks])+list(np.arange(4,60,6));modes=[]
    for df in starts:
        x=np.zeros(dim);x[0]=df
        fit=minimize(lambda v:-logtarget(v)[0],x,method='Nelder-Mead',bounds=[(1,60)]+[(None,None)]*(dim-1),
                     options={'maxiter':1000,'xatol':1e-6,'fatol':1e-7})
        if all(np.linalg.norm(fit.x-p[0])>.05 for p in modes):modes.append((fit.x,-fit.fun))
    best=max(p[1] for p in modes);modes=[p for p in modes if p[1]>best-25]
    locations=[];covariances=[];logweights=[]
    for loc,peak in modes:
        if loc[0]<1.005 or loc[0]>59.995:continue
        h=.002;H=np.empty((dim,dim));f0=-logtarget(loc)[0]
        for i in range(dim):
            ei=np.eye(dim)[i]*h
            H[i,i]=(-logtarget(loc+ei)[0]+2*logtarget(loc)[0]-logtarget(loc-ei)[0])/h**2
            for j in range(i):
                ej=np.eye(dim)[j]*h
                H[i,j]=H[j,i]=(-logtarget(loc+ei+ej)[0]+logtarget(loc+ei-ej)[0]+logtarget(loc-ei+ej)[0]-logtarget(loc-ei-ej)[0])/(4*h*h)
        if not np.all(np.isfinite(H)):continue
        eig,V=np.linalg.eigh(H)
        eig=np.maximum(eig,1e-3);C=(V/eig)@V.T
        locations.append(loc);covariances.append(C);logweights.append(peak+.5*np.linalg.slogdet(C)[1])
    if not locations:raise RuntimeError('importance proposal Hessian failed')
    mixture=np.exp(np.asarray(logweights)-logsumexp(logweights))
    counts=rng.multinomial(sample_count,mixture);chunks=[]
    for n,loc,C in zip(counts,locations,covariances):
        if n:chunks.append(np.asarray(multivariate_t.rvs(loc=loc,shape=C,df=5,size=n,random_state=rng)).reshape(n,dim))
    points=np.concatenate(chunks);rng.shuffle(points)
    logproposal=logsumexp(np.array([np.log(w)+multivariate_t.logpdf(points,loc=loc,shape=C,df=5) for w,loc,C in zip(mixture,locations,covariances)]),axis=0)
    logweight=logtarget(points)-logproposal;weights=np.exp(logweight-logsumexp(logweight))
    def quantile(x,w):
        order=np.argsort(x);x=x[order];w=w[order]/w.sum();return np.interp([.025,.5,.975],np.cumsum(w)-.5*w,x)
    interval=quantile(points[:,0],weights);half=len(points)//2
    discrepancy=float(max(abs(quantile(points[:half,0],weights[:half])-quantile(points[half:,0],weights[half:]))))
    ess=float(1/np.sum(weights**2));adequate=ess>=1000 and discrepancy<.06
    return {'interval':interval,'importance_ess':ess,'half_quantile_difference_mm':discrepancy,'integration_adequate':adequate,
            'proposal_mode_locations':[p.tolist() for p in locations],'sample_count':sample_count}
