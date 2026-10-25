import numpy as np
import pytest
from scipy.stats import multivariate_normal,invgamma,kstest,truncnorm,norm
from step6_recursive_bayes.likelihood import GaussianLikelihood,shrink_covariance
from step6_recursive_bayes.priors import *
from step6_recursive_bayes.samplers import *
from step6_recursive_bayes.inference import Posterior
from step6_recursive_bayes.calibration import fit_reference_response,load_calibration
from step6_recursive_bayes.operators import make_operator

def test_normalized_density():
    y=np.array([.2,-.3,1]);C=np.array([[2,.2,0],[.2,1,.1],[0,.1,3]])
    lik=GaussianLikelihood(y,C,lambda t:np.array([t[0],0,0]))
    for s2 in [.2,1.,3.]:assert np.isclose(lik([.4],s2),multivariate_normal.logpdf(y,mean=[.4,0,0],cov=s2*C))

def test_tempered_ig_condition():
    for T in [1,2,20]:
        a,b=noise_conditional(3,2,14,7,T);s=np.geomspace(.1,10,99)
        direct=np.array([log_s2_prior(v,3,2)+(-7*np.log(v)-7/(2*v))/T for v in s])
        conditional=invgamma.logpdf(s,a=a,scale=b)
        assert np.ptp(direct-conditional)<1e-12

@pytest.mark.parametrize('kind',['uniform','truncated_normal','bimodal'])
def test_slice_known_distributions(kind):
    rng=np.random.default_rng(81)
    if kind=='uniform':bounds=(0,1);target=lambda x:0.;cdf=lambda x:np.clip(x,0,1);width=.2;current=.5
    elif kind=='truncated_normal':bounds=(-1,2);target=lambda x:-.5*x*x;cdf=lambda x:truncnorm.cdf(x,-1,2);width=.8;current=0.
    else:bounds=(-12,12);target=lambda x:np.logaddexp(norm.logpdf(x,-3,1),norm.logpdf(x,3,1))-np.log(2);cdf=lambda x:.5*norm.cdf(x,-3,1)+.5*norm.cdf(x,3,1);width=6;current=0.
    samples=[]
    for i in range(18000):
        current=slice_update(target,current,width,bounds,rng)
        if i>=2000:samples.append(current)
    assert kstest(samples,cdf).statistic<.04

def test_calibration_known_references():
    rng=np.random.default_rng(2);x=rng.normal(size=(4,6,96))+1j*rng.normal(size=(4,6,96));s=np.exp(.01j*np.arange(96));l=np.full(96,.1+.2j)
    y=l+s*x;S,L,d=fit_reference_response(y,x,[0,1,2],[3])
    assert np.allclose(S[0],s) and np.allclose(L[0],l) and d['heldout_relative_error']<1e-14
    with pytest.raises(ValueError):fit_reference_response(y,np.ones_like(x),[0,1],[2])
    with pytest.raises(FileNotFoundError,match='calibration_missing'):load_calibration(None,make_operator())

def test_covariance_shrinkage():
    x=np.random.default_rng(1).normal(size=(5,14));C,details=shrink_covariance(x)
    assert np.min(np.linalg.eigvalsh(C))>0 and details['shrinkage']==.2

def test_swap_formula():assert np.isclose(swap_log_accept(1,4,-10,-30),-15)

def test_checkpoint_exact_resume(tmp_path):
    lik=GaussianLikelihood(np.array([1.,2.]),np.eye(2),lambda t:np.repeat(t[0],2))
    p=Posterior(lik,lambda t:0. if abs(t[0])<=10 else -np.inf,[[-10,10]])
    settings={'n_temperatures':2,'max_temperature':4,'warmup':10,'draws':30,'widths':[1.],'checkpoint_every':5,'stop_after':20,
              'global_proposal':{'means':[[1.]],'covariances':[[[1.]]],'weights':[1.]}}
    path=tmp_path/'checkpoint.npz';sample_pt(p,[[0],[2]],settings,2,path)
    # stop_after is a debug interrupt trigger, not part of target; same config resume passes it once.
    resumed=sample_pt(p,[[0],[2]],settings,2,path,True)
    full=sample_pt(p,[[0],[2]],dict(settings,stop_after=10000),2)
    assert np.array_equal(resumed['samples'],full['samples']) and np.array_equal(resumed['s2'],full['s2'])

def test_batch_forward_matches_scalar():
    from step6_recursive_bayes.io import load_config
    from step6_recursive_bayes.simulate import build_context
    from step6_recursive_bayes.likelihood import predict_many,predict_observation
    from step6_recursive_bayes.operators import project_observation
    from pathlib import Path
    c=load_config(Path(__file__).resolve().parents[1]/'configs/step6_synthetic_uncertain.json')
    ctx=build_context(c,True,True);ts=np.array([[1.4,15,.1,-.2,.2],[2,20,-.1,.3,-1]])
    assert np.allclose(predict_many(ts,ctx),[project_observation(predict_observation(t,ctx),ctx.operator.Q) for t in ts],atol=1e-13)

def test_checkpoint_before_warmup(tmp_path):
    lik=GaussianLikelihood(np.array([1.,2.]),np.eye(2),lambda t:np.repeat(t[0],2))
    p=Posterior(lik,lambda t:0. if abs(t[0])<=10 else -np.inf,[[-10,10]])
    settings={'n_temperatures':2,'max_temperature':4,'warmup':10,'draws':20,'widths':[1.],'stop_after':5}
    path=tmp_path/'early.npz';early=sample_pt(p,[[0],[2]],settings,31,path)
    assert early['samples'].shape==(2,0,1)
    recovered=sample_pt(p,[[0],[2]],settings,31,path,True)
    full=sample_pt(p,[[0],[2]],dict(settings,stop_after=999),31)
    assert np.array_equal(recovered['samples'],full['samples'])

def test_rare_separated_mode_is_reported():
    from step6_recursive_bayes.diagnostics import modes_1d
    rng=np.random.default_rng(4);x=np.r_[rng.normal(8.4,.02,9900),rng.normal(15,.03,100)]
    modes=modes_1d(x)
    assert len(modes)==2 and np.isclose(modes[1]['estimated_basin_probability'],.01)
