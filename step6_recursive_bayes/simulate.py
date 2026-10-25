import numpy as np
from .types import Calibration,PosteriorContext
from .materials import load_material_model,cole_cole
from .operators import make_operator,project_observation
from .likelihood import predict_observation,context_likelihood,record_covariance,configure_long_convolution
from .priors import log_theta_prior
from .inference import Posterior
from .io import resolve

def synthetic_system(operator):
    fb=operator.f_full_hz-operator.fc_hz
    s=np.exp(-.5*(fb/1.0e9)**2)*np.exp(-2j*np.pi*fb*3*operator.dt_s)*np.exp(.2j)
    return Calibration(operator.f_full_hz,s[None].astype(np.complex128),np.zeros((1,operator.nfft),complex),
                       {'synthetic':True,'independent_reference':False,'reference_plane':'synthetic_plane','method':'known generating function'})

def build_context(config,uncertain=None,delay=False):
    op=make_operator();model=load_material_model(resolve(config,config['materials']['ascii_path']))
    base={k:cole_cole(op.f_full_hz,p) for k,p in model.parameters.items()}
    if uncertain is None:uncertain=config['materials']['mode']=='fat_uncertain'
    bounds=[config.get('model',{}).get('d_skin_bounds_mm',[.5,3.5]),config.get('model',{}).get('d_fat_bounds_mm',[1.,60.])]
    if uncertain:bounds.extend([[-np.inf,np.inf],[-np.inf,np.inf]])
    if delay:bounds.append([-np.inf,np.inf])
    windows=config.get('synthetic',{}).get('windows',3);sigma=config.get('synthetic',{}).get('sigma',.003)
    C=sigma**2*np.eye(14);cov=record_covariance(C,windows)
    context=PosteriorContext(op,synthetic_system(op),base,np.zeros((windows,14)),cov,np.asarray(bounds),
                            tuple(config['materials'].get(k,v) for k,v in [('u_eps_sd',.1),('u_loss_sd',.2)]),
                            calibration_delay_sd_s=15e-12 if delay else 0.)
    configure_long_convolution(context,model,config.get('model',{}).get('linear_convolution_factor',8))
    context.alpha0=config.get('noise',{}).get('alpha0',3.)
    context.beta0=config.get('noise',{}).get('beta0',2.)
    return context

def simulate_record(context,truth,seed,noiseless=False):
    rng=np.random.default_rng(seed);y=predict_observation(truth,context);z=project_observation(y,context.operator.Q)
    mean=np.broadcast_to(z,context.observed.shape).copy()
    noise=np.zeros(mean.size) if noiseless else rng.multivariate_normal(np.zeros(mean.size),context.covariance)
    context.observed=mean+noise.reshape(mean.shape)
    return y

def make_posterior(context):
    delay=context.calibration_delay_sd_s>0
    return Posterior(context_likelihood(context),lambda t:log_theta_prior(t,context.bounds,context.material_sds,delay),
                     context.bounds,context.alpha0,context.beta0)
