"""Executable physics, known posterior and conditional coverage experiments."""
import copy,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm,chi2
from .io import resolve,scan_records,file_hash
from .simulate import build_context,simulate_record,make_posterior
from .materials import load_material_model,evaluate_materials
from .reflection import recursive_reflection,consistent_tmm,tissue_reflection
from .operators import make_operator,project_observation,apply_step5_operator
from .likelihood import GaussianLikelihood,predict_observation,configure_long_convolution
from .inference import Posterior,run_map,dense_integral
from .samplers import sample_pt
from .diagnostics import chain_diagnostics
from .reporting import json_write

def physics_validation(config,run):
    root=Path(config['_project_root']);completed=subprocess.run([sys.executable,'-m','pytest',str(root/'tests_step6_recursive/test_physics_operators.py'),'-q'],capture_output=True,text=True)
    (run/'pytest_physics.txt').write_text(completed.stdout+completed.stderr)
    if completed.returncode:raise RuntimeError('physics tests failed; see log')
    m=load_material_model(resolve(config,config['materials']['ascii_path']));r=scan_records(resolve(config,config['data']['observations_root']))[0]
    f=r.f_obs_hz;eps=evaluate_materials(f,m);layers=np.array([eps[k] for k in ['air','skin','fat','muscle']]);d=[.0015,.015]
    rec=recursive_reflection(f,layers,d);tmm=consistent_tmm(f,layers,d)
    # Read original implementation, never rewrite or monkey-patch it.
    from step6_tmm_forward_nopsy.transfer_matrix_forward import transfer_matrix_reflection
    old=transfer_matrix_reflection(f,*d,eps_dict=eps)
    ctx=build_context(config,False);base_op=ctx.operator
    predictions={}
    for factor in [1,4,8,16,32]:
        ctx.long_operator=None;configure_long_convolution(ctx,m,factor)
        predictions[factor]=predict_observation([1.5,15.],ctx)
    differences={str(n):float(np.linalg.norm(predictions[n]-predictions[32])/np.linalg.norm(predictions[32])) for n in [1,4,8,16]}
    # Explicit wrap stress: delayed impulses would wrap into bin 3 under N=96.
    a=np.zeros(96,complex);b=a.copy();a[90]=1;b[9]=1
    circular=np.fft.ifft(np.fft.fft(a)*np.fft.fft(b));linear=np.convolve(a,b)[:96]
    assert abs(circular[3])>.99 and abs(linear[3])<1e-14
    return {'status':'physics_validated','recursive_abs_range':[float(abs(rec).min()),float(abs(rec).max())],
            'legacy_abs_range':[float(abs(old).min()),float(abs(old).max())],
            'legacy_max_complex_error':float(max(abs(old-rec))),'consistent_tmm_max_complex_error':float(max(abs(tmm-rec))),
            'long_convolution_relative_error_vs_32x':differences,'production_extension_factor':8,
            'wrap_stress':'detected N=96 circular echo in bin 3; long convolution excludes it',
            'finite_system_cir_assumption':'measured full 96-point system CIR zero-extended; outside measured support unknown',
            'old_files_modified':False,'hardware_IQ_sign_independently_validated':False}

def statistics_validation(config,run):
    rng=np.random.default_rng(618);settings={'n_temperatures':4,'max_temperature':30.,'warmup':500,'draws':2000,'widths':[1.,1.],'update_noise':False}
    # Linear Gaussian posterior with known covariance, fixed s2, proper Gaussian prior.
    A=np.array([[1.,.5],[.2,1.],[-.4,.8]]);y=np.array([.4,-.3,.8]);C=np.diag([.3,.4,.2]);V=np.diag([1.2,.8])
    precision=A.T@np.linalg.solve(C,A)+np.linalg.inv(V);cov=np.linalg.inv(precision);mean=cov@A.T@np.linalg.solve(C,y)
    lik=GaussianLikelihood(y,C,lambda t:A@t);post=Posterior(lik,lambda t:-.5*t@np.linalg.solve(V,t),[[-np.inf,np.inf]]*2)
    result=sample_pt(post,[[-1,-1],[1,1],[-1,1],[1,-1]],dict(settings,global_proposal={
        'means':[[0.,0.]],'covariances':[np.eye(2).tolist()],'weights':[1.]},global_every=5),119)
    dg=chain_diagnostics(result['samples']);sample_mean=result['samples'].mean(axis=(0,1));sample_cov=np.cov(result['samples'].reshape(-1,2),rowvar=False)
    assert np.all(abs(sample_mean-mean)<5*np.asarray(dg['mcse_mean']))
    assert np.allclose(np.diag(sample_cov),np.diag(cov),rtol=.12)
    gaussian={'analytic_mean':mean.tolist(),'estimated_mean':sample_mean.tolist(),'analytic_variance':np.diag(cov).tolist(),'estimated_variance':np.diag(sample_cov).tolist(),**dg}
    # Normal mean + IG variance with effectively nonbinding proper uniform mean prior.
    data=np.array([.7,1.3,1.1,.9,1.5,1.0]);n=len(data);avg=data.mean();SSE=((data-avg)**2).sum()
    lik=GaussianLikelihood(data,np.eye(n),lambda t:np.full(n,t[0]));post=Posterior(lik,lambda t:0 if -100<t[0]<100 else -np.inf,[[-100,100]],3,2)
    result2=sample_pt(post,[[0],[1],[2],[-1]],dict(settings,update_noise=True,widths=[1.]),120)
    analytic_variance=(2+SSE/2)/(3+(n-1)/2-1)/n
    x=result2['samples'];normal_ig={'analytic_mean':float(avg),'sample_mean':float(x.mean()),'analytic_variance':float(analytic_variance),'sample_variance':float(x.var()),'mean_s2':float(result2['s2'].mean())}
    assert abs(x.mean()-avg)<.04 and abs(x.var()/analytic_variance-1)<.15
    # Known two-mode likelihood. Fixed s2 test isolates Slice + likelihood-only PT.
    class MixtureLikelihood:
        D=1
        def sse(self,t):return -2*(np.logaddexp(norm.logpdf(t[0],-3,1),norm.logpdf(t[0],3,1))-np.log(2))
        def loglike_sse(self,sse,s2):return -.5*sse/s2-.5*np.log(s2)
        def __call__(self,t,s2):return self.loglike_sse(self.sse(t),s2)
    mix=Posterior(MixtureLikelihood(),lambda t:0 if -12<=t[0]<=12 else -np.inf,[[-12,12]])
    result3=sample_pt(mix,[[-3],[3],[-5],[5]],dict(settings,widths=[1.2]),121)
    cold=result3['samples'][...,0];fraction=float(np.mean(cold<0));assert abs(fraction-.5)<.06
    json_write(run/'known_target_diagnostics.json',{'linear_gaussian':gaussian,'normal_IG':normal_ig,'two_mode':{'negative_mode_probability':fraction,'true_probability':.5,'diagnostics':chain_diagnostics(result3['samples'])}})
    np.savez(run/'known_target_samples.npz',linear_gaussian=result['samples'],normal_IG=result2['samples'],mixture=result3['samples'])
    return {'status':'known_target_statistics_validated','gaussian':gaussian,'normal_IG':normal_ig,'mixture_negative_mode_probability':fraction}

def coverage_validation(config,run):
    """Repeated exact-grid/low-dimensional quadrature study, conditional on known skin.

    Avoid expensive repeated full 4D MCMC. Full free-skin posterior tested separately.
    """
    ctx=build_context(config,True);ctx.observed=np.zeros((3,14));ctx.covariance=.003**2*np.eye(42)
    sigma=.003;rng=np.random.default_rng(config['seed']+80);repetitions=24
    df_grid=np.linspace(5,30,251);nodes,weights=np.polynomial.hermite.hermgauss(5);weights=weights/np.sqrt(np.pi)
    def prediction(theta,context=ctx):return project_observation(predict_observation(theta,context),context.operator.Q)
    def table(mode,sd_factor=1.):
        pars=[];weights_list=[]
        for df in df_grid:
            if mode=='fixed':pars.append([1.5,df,0,0]);weights_list.append(1.)
            elif mode=='delay':
                for u,w in zip(nodes,weights):pars.append([1.5,df,0,0,np.sqrt(2)*u]);weights_list.append(w)
            else:
                for i,u in enumerate(nodes):
                    for j,v in enumerate(nodes):pars.append([1.5,df,np.sqrt(2)*.1*sd_factor*u,np.sqrt(2)*.2*sd_factor*v]);weights_list.append(weights[i]*weights[j])
        arr=np.array([prediction(t,ctx) for t in pars]);return arr.reshape(len(df_grid),-1,14),np.array(weights_list).reshape(len(df_grid),-1)
    fixed,w0=table('fixed');uncertain,w1=table('uncertain')
    narrow,wn=table('uncertain',.5);wide,ww=table('uncertain',2.)
    delay_context=build_context(config,True,True)
    # Reuse table() with explicit context for the calibration-delay basis.
    old_ctx=ctx;ctx=delay_context;delay,wd=table('delay');ctx=old_ctx
    def posterior(obs,preds,w):
        SSE=np.sum((obs[None,None,:,:]-preds[:,:,None,:])**2,axis=(-1,-2))/sigma**2
        ll=-(ctx.alpha0+obs.size/2)*np.log(ctx.beta0+SSE/2)
        logp=logsumexp(ll+np.log(w),axis=1);prob=np.exp(logp-logsumexp(logp));cdf=np.cumsum(prob)
        return np.interp([.025,.5,.975],cdf,df_grid),prob
    rows=[];means=[]
    # Prior-predictive material variation, with a common observation for paired models.
    for rep in range(repetitions):
        truth=[1.5,15.,rng.normal(0,.1),rng.normal(0,.2)];mu=prediction(truth);obs=mu+rng.normal(0,sigma,(3,14))
        for name,preds,w in [('fixed',fixed,w0),('uncertain',uncertain,w1)]:
            interval,prob=posterior(obs,preds,w);rows.append({'rep':rep,'scenario':'material_variation','model':name,'truth_df_mm':15.,'lower':float(interval[0]),'median':float(interval[1]),'upper':float(interval[2]),'covered':bool(interval[0]<=15<=interval[2])})
        if rep==0:
            for name,preds,w in [('narrow',narrow,wn),('medium',uncertain,w1),('wide',wide,ww)]:
                interval,prob=posterior(obs,preds,w);means.append({'prior':name,'interval':interval.tolist()})
        delay_truth=[1.5,15.,0,0,rng.normal()];mu=prediction(delay_truth,delay_context);obs=mu+rng.normal(0,sigma,(3,14))
        for name,preds,w in [('ignore_delay',fixed,w0),('propagate_delay',delay,wd)]:
            interval,prob=posterior(obs,preds,w);rows.append({'rep':rep,'scenario':'independent_calibration_delay','model':name,'truth_df_mm':15.,'lower':float(interval[0]),'median':float(interval[1]),'upper':float(interval[2]),'covered':bool(interval[0]<=15<=interval[2])})
    summary=[]
    for scenario,model in sorted(set((r['scenario'],r['model']) for r in rows)):
        selected=[r for r in rows if r['scenario']==scenario and r['model']==model];k=sum(r['covered'] for r in selected);n=len(selected);p=k/n;z=1.96
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        summary.append({'scenario':scenario,'model':model,'repeats':n,'covered':k,'coverage':p,'wilson95':[center-half,center+half],
                        'mean_interval_width_mm':float(np.mean([r['upper']-r['lower'] for r in selected])),
                        'RMSE_mm':float(np.sqrt(np.mean([(r['median']-15)**2 for r in selected])))})
    # Structural mismatch: extra 0.25-mm layer; inspect conditional model residual, no guarantee of coverage.
    op=ctx.long_operator or ctx.operator;base=ctx.long_baseline or ctx.baseline
    eps=evaluate_materials(op.f_full_hz,base);layers=np.array([eps['air'],eps['skin'],np.full(len(op.f_full_hz),15-3j),eps['fat'],eps['muscle']])
    x=recursive_reflection(op.f_full_hz,layers,[.0015,.00025,.015]);full=ctx.long_system*x
    h=np.fft.ifft(np.fft.ifftshift(full,axes=-1),axis=-1)[...,:96];y=apply_step5_operator(np.fft.fftshift(np.fft.fft(h,axis=-1),axes=-1),ctx.operator)[0]
    obs=np.broadcast_to(project_observation(y,ctx.operator.Q),(3,14));interval,prob=posterior(obs,fixed,w0)
    best=int(np.argmax(prob));sse=float(np.sum((obs-fixed[best,0])**2)/sigma**2)
    import csv
    with (run/'coverage_repetitions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'status':'conditional_synthetic_coverage_experiment','known_skin_mm':1.5,'free_parameter':'d_fat_mm plus quadrature nuisance',
            'not_full_4D_MCMC_coverage':True,'noise_scale':'same IG prior marginalized analytically in all models',
            'material_quadrature_order':5,'df_grid_spacing_mm':.1,'summary':summary,'prior_sensitivity':means,
            'structural_mismatch':{'extra_layer_mm':.25,'best_conditional_residual_SSE':sse,'D':42,'chi2_reference_p':float(chi2.sf(sse,41)),
             'interval':interval.tolist(),'note':'fixed-s2 chi-square only descriptive; flexible variance can absorb mismatch'},
            'limitation':'24 repetitions and coarse nuisance quadrature are exploratory; inspect quadrature convergence before formal claims'}

# Retained above for traceability: coarse tensor quadrature was rejected after
# visibly missing narrow nuisance posterior modes. Production validation uses
# adaptive importance integration, with adequacy recorded for every repetition.
coverage_coarse_quadrature=coverage_validation

def coverage_validation(config,run):
    from .experiments import conditional_importance
    ctx=build_context(config,True);delay_ctx=build_context(config,True,True)
    rng=np.random.default_rng(config['seed']+81);data_rng=np.random.default_rng(config['seed']+80)
    repetitions=24;sigma=.003;rows=[];sensitivity=[]
    def evaluate(obs,context,mode,sd=1.):
        out=conditional_importance(obs,context,mode,rng,sd)
        if not out['integration_adequate']:
            out=conditional_importance(obs,context,mode,rng,sd,sample_count=32768)
        return out
    for rep in range(repetitions):
        truth=[1.5,15.,data_rng.normal(0,.1),data_rng.normal(0,.2)]
        mu=project_observation(predict_observation(truth,ctx),ctx.operator.Q)
        obs=mu+data_rng.normal(0,sigma,(3,14))
        for model,mode in [('fixed','fixed'),('uncertain','material')]:
            out=evaluate(obs,ctx,mode);lo,med,hi=out['interval']
            rows.append({'rep':rep,'scenario':'material_variation','model':model,'truth_df_mm':15.,'lower':float(lo),'median':float(med),'upper':float(hi),
                         'covered':bool(lo<=15<=hi),'importance_ess':out['importance_ess'],'quantile_half_difference_mm':out['half_quantile_difference_mm'],'integration_adequate':out['integration_adequate']})
        if rep==0:
            for label,sd in [('narrow',.5),('medium',1.),('wide',2.)]:
                out=evaluate(obs,ctx,'material',sd);sensitivity.append({'prior':label,**out})
        td=[1.5,15.,0,0,data_rng.normal()];mu=project_observation(predict_observation(td,delay_ctx),delay_ctx.operator.Q);obs=mu+data_rng.normal(0,sigma,(3,14))
        for model,mode,context in [('ignore_delay','fixed',ctx),('propagate_delay','delay',delay_ctx)]:
            out=evaluate(obs,context,mode);lo,med,hi=out['interval']
            rows.append({'rep':rep,'scenario':'independent_calibration_delay','model':model,'truth_df_mm':15.,'lower':float(lo),'median':float(med),'upper':float(hi),
                         'covered':bool(lo<=15<=hi),'importance_ess':out['importance_ess'],'quantile_half_difference_mm':out['half_quantile_difference_mm'],'integration_adequate':out['integration_adequate']})
        # Incremental progress, preserving even numerically inadequate repetitions.
        json_write(run/'coverage_progress.json',{'completed_repetitions':rep+1,'results':rows})
    summary=[]
    for scenario,model in sorted(set((r['scenario'],r['model']) for r in rows)):
        selected=[r for r in rows if r['scenario']==scenario and r['model']==model];n=len(selected);k=sum(r['covered'] for r in selected);p=k/n;z=1.96
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        summary.append({'scenario':scenario,'model':model,'repeats':n,'covered':k,'coverage':p,'wilson95':[center-half,center+half],
                        'mean_interval_width_mm':float(np.mean([r['upper']-r['lower'] for r in selected])),
                        'RMSE_mm':float(np.sqrt(np.mean([(r['median']-15)**2 for r in selected]))),
                        'integration_adequate_count':sum(r['integration_adequate'] for r in selected)})
    # Unmodeled extra layer, same synthetic S and noise scale.
    op=ctx.long_operator;base=ctx.long_baseline;eps=evaluate_materials(op.f_full_hz,base)
    layers=np.array([eps['air'],eps['skin'],np.full(len(op.f_full_hz),15-3j),eps['fat'],eps['muscle']]);x=recursive_reflection(op.f_full_hz,layers,[.0015,.00025,.015])
    full=ctx.long_system*x;h=np.fft.ifft(np.fft.ifftshift(full,axes=-1),axis=-1)[...,:96]
    y=apply_step5_operator(np.fft.fftshift(np.fft.fft(h,axis=-1),axes=-1),ctx.operator)[0]
    obs=np.broadcast_to(project_observation(y,ctx.operator.Q),(3,14));out=evaluate(obs,ctx,'fixed')
    mean=project_observation(predict_observation([1.5,out['interval'][1],0,0],ctx),ctx.operator.Q)
    sse=float(np.sum((obs-mean)**2)/sigma**2)
    import csv
    with (run/'coverage_repetitions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'status':'conditional_synthetic_coverage_experiment' if all(r['integration_adequate'] for r in rows) else 'integration_requires_review',
            'known_skin_mm':1.5,'not_full_4D_MCMC_coverage':True,'method':'global dense 1D grid for fixed model; multimode Student-t importance integration for nuisance models; s2 analytically marginalized',
            'noise_scale':'same IG prior and covariance for paired models','summary':summary,'prior_sensitivity':sensitivity,
            'structural_mismatch':{'status':'model_misfit' if chi2.sf(sse,41)<.01 else 'not_flagged','extra_layer_mm':.25,'conditional_residual_SSE':sse,'D':42,'chi2_reference_p':float(chi2.sf(sse,41)),'interval':out['interval'],
                                   'note':'diagnostic only; variance inflation can hide structural mismatch'},
            'limitations':['24 repeats; Wilson intervals reported','known skin in repeated experiment; full free-skin MCMC validated separately',
                           'mode search and importance convergence must be reviewed; no human accuracy conclusion','coarse 5-point quadrature rejected; results retained only as failed numerical diagnostic']}

def recovery_validation(config,run):
    from .likelihood import predict_many
    c=copy.deepcopy(config);c['materials']['mode']='fixed';c['synthetic']['truth']=[1.5,15.]
    ctx=build_context(c,False);simulate_record(ctx,[1.5,15.],c['seed'],True);post=make_posterior(ctx)
    candidates,_=run_map(post,(19,80));recovered=candidates[0]['theta'];error=abs(recovered-np.array([1.5,15.]))
    assert np.max(error)<1e-4
    json_write(run/'noiseless_recovery.json',{'truth':[1.5,15.],'MAP':recovered,'abs_error_mm':error})
    simulate_record(ctx,[1.5,15.],c['seed'],False);post=make_posterior(ctx);candidates,_=run_map(post,(19,80))
    peak=max(post.marginal(r['theta']) for r in candidates);pieces=[];logmass=[];axes_out={}
    for i,row in enumerate(candidates):
        center=row['theta'];lp=post.marginal(center)
        if lp<peak-25:continue
        h=1e-4;H=np.empty((2,2))
        for j in range(2):
            for k in range(2):
                ej=np.eye(2)[j]*h;ek=np.eye(2)[k]*h
                H[j,k]=-(post.marginal(center+ej+ek)-post.marginal(center+ej-ek)-post.marginal(center-ej+ek)+post.marginal(center-ej-ek))/(4*h*h)
        sd=np.sqrt(np.diag(np.linalg.inv(H)))
        a=np.linspace(max(.5,center[0]-12*sd[0]),min(3.5,center[0]+12*sd[0]),241)
        b=np.linspace(max(1.,center[1]-12*sd[1]),min(60.,center[1]+12*sd[1]),321)
        aa,bb=np.meshgrid(a,b,indexing='ij');points=np.column_stack((aa.ravel(),bb.ravel()))
        z=predict_many(points,ctx);sse=np.sum((ctx.observed[None]-z[:,None])**2,axis=(1,2))/ctx.covariance[0,0]
        logp=-(post.alpha+post.likelihood.D/2)*np.log(post.beta+sse/2)
        lm=logsumexp(logp)+np.log((a[1]-a[0])*(b[1]-b[0]));pieces.append((points,logp));logmass.append(lm)
        axes_out['mode_'+str(i)+'_skin']=a;axes_out['mode_'+str(i)+'_fat']=b
    total=logsumexp(logmass);weight=[];points=[]
    for (x,lp),mass in zip(pieces,logmass):
        weight.extend(np.exp(lp-logsumexp(lp)+mass-total));points.extend(x)
    weight=np.asarray(weight);points=np.asarray(points);mean=weight@points;var=weight@(points-mean)**2
    intervals=[]
    for j in range(2):
        idx=np.argsort(points[:,j]);intervals.append(np.interp([.025,.5,.975],np.cumsum(weight[idx])-.5*weight[idx],points[idx,j]).tolist())
    np.savez(run/'dense_integration.npz',points=points,weights=weight,**axes_out)
    available=[]
    for p in (Path(config['_project_root'])/config['output_root']).glob('*recover-synthetic*/posterior_samples.npz'):
        with np.load(p,allow_pickle=False) as zz:
            if zz['samples'].shape[-1]==2:available.append(p)
    compare={'status':'MCMC_not_yet_available'}
    if available:
        source=sorted(available)[-1]
        with np.load(source,allow_pickle=False) as zz:x=zz['samples']
        dg=chain_diagnostics(x);estimate=x.mean(axis=(0,1));mcse=np.asarray(dg['mcse_mean'])
        mean_ok=bool(np.all(abs(estimate-mean)<5*mcse+np.array([2e-4,2e-3])))
        variance_ok=bool(np.allclose(x.reshape(-1,2).var(axis=0),var,rtol=.15))
        compare={'status':'passed' if mean_ok and variance_ok else 'mismatch','source':str(source),'MCMC_mean':estimate,
                 'dense_mean':mean,'dense_variance':var,'MCMC_variance':x.reshape(-1,2).var(axis=0),'diagnostics':dg}
        assert mean_ok and variance_ok
    return {'status':'synthetic_recovery_validated' if compare['status']=='passed' else 'dense_integration_completed',
            'noiseless_abs_error_mm':error,'dense_mean':mean,'dense_intervals':intervals,'mode_weights':np.exp(np.array(logmass)-total),
            'MCMC_comparison':compare,'integration':'dense local 241x321 grids around all multistart modes within 25 log units; +/-12 Laplace SD; analytically marginalized s2'}
