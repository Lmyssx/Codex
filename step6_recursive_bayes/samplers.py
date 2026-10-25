import copy,json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import logsumexp
from .priors import noise_conditional,log_s2_prior

def slice_update(log_target,current,width,bounds,rng,max_steps=100):
    lo,hi=bounds
    logp=log_target(current)
    if not np.isfinite(logp) or width<=0: raise ValueError('invalid Slice initial state/width')
    level=logp-rng.exponential();raw_left=current-rng.uniform()*width
    left=max(lo,raw_left);right=min(hi,raw_left+width)
    # Random finite stepping-out budget preserves detailed balance.
    j=int(rng.integers(max_steps));k=max_steps-1-j
    while j>0 and left>lo and log_target(left)>level:left=max(lo,left-width);j-=1
    while k>0 and right<hi and log_target(right)>level:right=min(hi,right+width);k-=1
    for _ in range(10000):
        proposal=rng.uniform(left,right)
        if log_target(proposal)>=level:return proposal
        if proposal<current:left=proposal
        else:right=proposal
    raise RuntimeError('Slice shrinkage failed')

def swap_log_accept(Tj,Tk,llj,llk):return (1/Tj-1/Tk)*(llk-llj)

def sample_pt(posterior,initials,settings,seed,checkpoint=None,resume=False):
    """Checkpoint includes retained cold history and exact RNG state; resume is bitwise reproducible."""
    rng=np.random.default_rng(seed);initials=np.asarray(initials,float)
    E=len(initials);K=settings['n_temperatures'];warm=settings['warmup'];draws=settings['draws']
    T=np.geomspace(1,settings['max_temperature'],K)
    widths=np.asarray(settings.get('widths',[.25,3.,.07,.12,1.])[:initials.shape[1]])
    import hashlib
    fingerprint=hashlib.sha256(json.dumps({'settings':settings,'bounds':posterior.bounds.tolist(),'seed':seed,
                           'target_id':settings.get('target_id','unspecified')},sort_keys=True).encode()).hexdigest()
    theta=np.repeat(initials[:,None,:],K,axis=1);s2=np.ones((E,K));start=0
    samples=[];scales=[];lls=[];lps=[];attempts=np.zeros(K-1,int);accepted=attempts.copy()
    global_proposal=settings.get('global_proposal')
    global_attempts=0;global_accepts=0
    def propose_global(current,temperature):
        weights=np.asarray(global_proposal['weights'])
        means=global_proposal['means'];covs=np.asarray(global_proposal['covariances'])*temperature
        chosen=rng.choice(len(weights),p=weights)
        candidate=np.atleast_1d(multivariate_t.rvs(loc=means[chosen],shape=covs[chosen],df=5,random_state=rng))
        def logq(t):return logsumexp([np.log(w)+multivariate_t.logpdf(t,loc=m,shape=C,df=5) for w,m,C in zip(weights,means,covs)])
        return candidate,logq(current)-logq(candidate)
    if resume:
        if checkpoint is None:raise ValueError('checkpoint path required')
        with np.load(checkpoint,allow_pickle=False) as z:
            if z['fingerprint'].item()!=fingerprint:raise ValueError('checkpoint configuration/target mismatch')
            theta=z['theta'];s2=z['s2'];start=int(z['iteration']);rng.bit_generator.state=json.loads(z['rng'].item())
            samples=list(z['samples']);scales=list(z['scales']);lls=list(z['lls']);lps=list(z['lps']);attempts=z['attempts'];accepted=z['accepted']
            global_attempts=int(z['global_attempts']) if 'global_attempts' in z else 0
            global_accepts=int(z['global_accepts']) if 'global_accepts' in z else 0
    def save(iteration):
        if checkpoint is None:return
        p=Path(checkpoint);tmp=p.with_suffix('.tmp.npz')
        np.savez(tmp,theta=theta,s2=s2,iteration=iteration,rng=json.dumps(rng.bit_generator.state),fingerprint=fingerprint,
                 samples=np.asarray(samples),scales=np.asarray(scales),lls=np.asarray(lls),lps=np.asarray(lps),attempts=attempts,accepted=accepted,
                 global_attempts=global_attempts,global_accepts=global_accepts,temperatures=T)
        tmp.replace(p)
    for iteration in range(start,warm+draws):
        ll=np.zeros((E,K))
        for e in range(E):
            for k in range(K):
                state=theta[e,k].copy()
                for i in rng.permutation(len(state)):
                    def target(value):
                        candidate=state.copy();candidate[i]=value;lp=posterior.logprior(candidate)
                        return lp+posterior.likelihood(candidate,s2[e,k])/T[k] if np.isfinite(lp) else -np.inf
                    state[i]=slice_update(target,state[i],widths[i],posterior.bounds[i],rng)
                theta[e,k]=state;sse=posterior.likelihood.sse(state)
                a,b=noise_conditional(posterior.alpha,posterior.beta,posterior.likelihood.D,sse,T[k])
                if settings.get('update_noise',True):s2[e,k]=1/rng.gamma(a,1/b)
                ll[e,k]=posterior.likelihood.loglike_sse(sse,s2[e,k])
                if global_proposal and iteration%settings.get('global_every',5)==0:
                    global_attempts+=1
                    candidate,proposal_ratio=propose_global(theta[e,k],T[k])
                    lp=posterior.logprior(candidate)
                    if np.isfinite(lp):
                        candidate_ll=posterior.likelihood(candidate,s2[e,k])
                        ratio=(candidate_ll-ll[e,k])/T[k]+lp-posterior.logprior(theta[e,k])+proposal_ratio
                        if np.log(rng.uniform())<min(0.,ratio):
                            theta[e,k]=candidate;ll[e,k]=candidate_ll;global_accepts+=1
            for k in range(iteration%2,K-1,2):
                attempts[k]+=1
                if np.log(rng.uniform())<min(0,swap_log_accept(T[k],T[k+1],ll[e,k],ll[e,k+1])):
                    theta[e,[k,k+1]]=theta[e,[k+1,k]];s2[e,[k,k+1]]=s2[e,[k+1,k]];ll[e,[k,k+1]]=ll[e,[k+1,k]];accepted[k]+=1
        if iteration>=warm:
            samples.append(theta[:,0].copy());scales.append(s2[:,0].copy());lls.append(ll[:,0].copy())
            lps.append(np.array([posterior.logprior(theta[e,0])+log_s2_prior(s2[e,0],posterior.alpha,posterior.beta) for e in range(E)]))
        if (iteration+1)%settings.get('checkpoint_every',250)==0:save(iteration+1)
        if settings.get('stop_after')==iteration+1:save(iteration+1);break
    else:save(warm+draws)
    cold=np.asarray(samples).transpose(1,0,2) if samples else np.empty((E,0,initials.shape[1]))
    return {'samples':cold,'s2':np.asarray(scales).T if scales else np.empty((E,0)),
            'loglike':np.asarray(lls).T,'logprior':np.asarray(lps).T,
            'temperatures':T,'swap_attempts':attempts,'swap_accepts':accepted,'warmup':warm,
            'global_attempts':global_attempts,'global_accepts':global_accepts}
