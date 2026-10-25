import numpy as np
from scipy.linalg import cholesky,solve_triangular,block_diag
from .materials import evaluate_materials
from .reflection import tissue_reflection
from .operators import apply_step5_operator,project_observation

def predict_observation(theta,context):
    t=np.asarray(theta);target_op=context.operator;op=context.long_operator or target_op
    baseline=context.long_baseline if context.long_operator is not None else context.baseline
    eps=evaluate_materials(op.f_full_hz,baseline,t[2:4] if len(t)>=4 else (0.,0.))
    x=tissue_reflection(op.f_full_hz,eps,t[0]*1e-3,t[1]*1e-3,context.g_air_m)
    S=context.long_system if context.long_operator is not None else context.calibration.system_response_full
    if context.calibration_delay_sd_s>0:
        # Delay coefficient is in units of its independently specified SD.
        S=S*np.exp(-2j*np.pi*(op.f_full_hz-op.fc_hz)*context.calibration_delay_sd_s*t[-1])
    offset=context.long_offset if context.long_operator is not None else context.calibration.residual_offset_full
    full=S*x+offset
    if context.long_operator is not None:
        # Finite measured system CIR, zero-extended; dispersive response on a longer grid.
        # Crop the first ORIGINAL 96 bins before the unchanged original-position gate.
        h=np.fft.ifft(np.fft.ifftshift(full,axes=-1),axis=-1)[...,:target_op.nfft]
        full=np.fft.fftshift(np.fft.fft(h,axis=-1),axes=-1)
    return apply_step5_operator(full,target_op)[0]

def configure_long_convolution(context,material_model,factor=8):
    from .operators import make_operator
    from .materials import cole_cole
    if factor<1 or int(factor)!=factor:raise ValueError('integer convolution extension required')
    if factor==1:return
    op=context.operator
    long=make_operator(op.fc_hz,op.bandwidth_hz,op.dt_s,op.nfft*factor,op.gate_bins)
    context.long_operator=long
    context.long_baseline={k:cole_cole(long.f_full_hz,p) for k,p in material_model.parameters.items()}
    def extend(a):
        h=np.fft.ifft(np.fft.ifftshift(a,axes=-1),axis=-1)
        return np.fft.fftshift(np.fft.fft(h,n=long.nfft,axis=-1),axes=-1)
    context.long_system=extend(context.calibration.system_response_full)
    context.long_offset=extend(context.calibration.residual_offset_full)

def shrink_covariance(samples,shrinkage=.2):
    x=np.asarray(samples,float)
    if x.ndim!=2 or len(x)<2 or not 0<shrinkage<=1: raise ValueError('invalid covariance samples/shrinkage')
    raw=np.cov(x,rowvar=False);scale=np.trace(raw)/len(raw)
    if not scale>0: raise ValueError('zero noise variance')
    C=(1-shrinkage)*raw+shrinkage*scale*np.eye(len(raw))
    return C,{'shrinkage':shrinkage,'eigenvalues':np.linalg.eigvalsh(C).tolist(),'sample_count':len(x),
              'note':'time/record correlations and background mean uncertainty require separate validation'}

def record_covariance(C,windows,rho=0.,shared_fraction=0.,indices=None):
    if not 0<=rho<1 or not 0<=shared_fraction<=1: raise ValueError('invalid temporal/shared noise')
    positions=np.arange(windows) if indices is None else np.asarray(indices)
    if positions.shape!=(windows,):raise ValueError('window index dimension')
    T=rho**abs(positions[:,None]-positions[None,:])
    return np.kron(T+shared_fraction*np.ones_like(T),C)

class GaussianLikelihood:
    def __init__(self,observed,covariance,predict):
        self.observed=np.asarray(observed,float).reshape(-1)
        self.L=cholesky(covariance,lower=True,check_finite=True)
        if self.L.shape!=(self.observed.size,self.observed.size): raise ValueError('covariance dimension')
        self.D=self.observed.size;self.logdet_half=float(np.log(np.diag(self.L)).sum());self.predict=predict
    def residual(self,theta):
        predicted=np.asarray(self.predict(theta),float).reshape(-1)
        if predicted.size!=self.D: raise ValueError('prediction dimension mismatch')
        return solve_triangular(self.L,self.observed-predicted,lower=True,check_finite=False)
    def sse(self,theta):
        e=self.residual(theta);return float(e@e)
    def loglike_sse(self,sse,s2):
        if not np.isfinite(s2) or s2<=0: return -np.inf
        return -.5*self.D*np.log(2*np.pi*s2)-self.logdet_half-.5*sse/s2
    def __call__(self,theta,s2): return self.loglike_sse(self.sse(theta),s2)

def context_likelihood(context):
    observed=np.asarray(context.observed)
    def prediction(theta):
        z=project_observation(predict_observation(theta,context),context.operator.Q)
        return np.broadcast_to(z,observed.shape)
    return GaussianLikelihood(observed,context.covariance,prediction)

def predict_many(theta,context,chunk_size=256):
    """Batch forward evaluator for numerical integration, same physics/operator as scalar."""
    values=np.atleast_2d(theta);op=context.long_operator or context.operator
    base=context.long_baseline if context.long_operator is not None else context.baseline
    S=context.long_system if context.long_operator is not None else context.calibration.system_response_full
    offset=context.long_offset if context.long_operator is not None else context.calibration.residual_offset_full
    f=op.f_full_hz;skin=np.sqrt(base['skin']);muscle=np.sqrt(base['muscle']);a=(1/skin-1)/(1/skin+1);out=[]
    for start in range(0,len(values),chunk_size):
        t=values[start:start+chunk_size];ue=t[:,2:3] if t.shape[1]>=4 else 0.;ul=t[:,3:4] if t.shape[1]>=4 else 0.
        fat=np.sqrt(np.exp(ue)*base['fat'].real+1j*np.exp(ul)*base['fat'].imag)
        r12=(skin-fat)/(skin+fat);r23=(fat-muscle)/(fat+muscle)
        q=r23*np.exp(-4j*np.pi*f/299792458*fat*t[:,1:2]*1e-3)
        r=(r12+q)/(1+r12*q)
        q=r*np.exp(-4j*np.pi*f/299792458*skin*t[:,0:1]*1e-3)
        x=(a+q)/(1+a*q)*np.exp(-4j*np.pi*f/299792458*context.g_air_m)
        full=S*x
        if context.calibration_delay_sd_s>0:
            full*=np.exp(-2j*np.pi*(f-op.fc_hz)*context.calibration_delay_sd_s*t[:,-1:])
        full+=offset
        if context.long_operator is not None:
            h=np.fft.ifft(np.fft.ifftshift(full,axes=-1),axis=-1)[...,:context.operator.nfft]
            full=np.fft.fftshift(np.fft.fft(h,axis=-1),axes=-1)
        out.append(project_observation(apply_step5_operator(full,context.operator),context.operator.Q))
    return np.concatenate(out)
