import csv,json,platform,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .io import file_hash
from .diagnostics import chain_diagnostics,interval_summary,modes_1d

def json_write(path,data):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf-8')

def new_run(config,command):
    root=Path(config['_project_root'])/config['output_root'];root.mkdir(exist_ok=True,parents=True)
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+command+'_'+uuid.uuid4().hex[:6]
    run=root/name;run.mkdir()
    json_write(run/'config_resolved.json',config)
    import scipy
    import importlib.metadata as im
    versions={p:im.version(p) for p in ['numpy','scipy','arviz-stats','pytest']}
    inputs=[]
    for key in ['ascii_path']:
        p=Path(config['_project_root'])/config['materials'][key]
        if p.exists():inputs.append({'path':str(p),'sha256':file_hash(p)})
    observations=Path(config['_project_root'])/config['data']['observations_root']
    for p in sorted(observations.rglob('*_observations.npz')):
        inputs.append({'path':str(p),'sha256':file_hash(p)})
    source_hashes={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    json_write(run/'manifest.json',{'schema_version':1,'command':command,'seed':config['seed'],'python':sys.version,
                                   'platform':platform.platform(),'versions':versions,'units':{'core':'Hz,m,s,S/m','output_thickness':'mm'},
                                   'inputs':inputs,'source_hashes':source_hashes,'model_version':'recursive_v0.1.0','operator_version':'step5_original_gate_numpy_fft',
                                   'config_sha256':file_hash(run/'config_resolved.json'),'blas':str(np.__config__.CONFIG.get('Build Dependencies',{}).get('blas',{}))})
    return run

def save_map(run,candidates,grid):
    names=['d_skin_mm','d_fat_mm','u_eps_fat','u_loss_fat','calibration_delay_standardized'][:len(candidates[0]['theta'])]
    with (run/'map_candidates.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=names+['s2','logposterior','loglike','success','boundary']);writer.writeheader()
        for row in candidates:writer.writerow(dict(zip(names,row['theta']))|{k:row[k] for k in ['s2','logposterior','loglike','success','boundary']})
    np.savez(run/'map_grid.npz',**grid)

def save_posterior(run,result,context,posterior,map_candidates=None,smoke=False):
    x=result['samples'];names=['d_skin_mm','d_fat_mm']
    if x.shape[-1]>=4:names+=['u_eps_fat','u_loss_fat']
    if context.calibration_delay_sd_s>0:names+=['calibration_delay_standardized']
    np.savez(run/'posterior_samples.npz',**result,param_names=np.array(names),param_units=np.array(['mm','mm']+['dimensionless']*(len(names)-2)))
    complete=np.concatenate((x,result['s2'][...,None]),axis=-1)
    dg=chain_diagnostics(complete);dg['diagnostic_param_names']=names+['s2'];dg['smoke']=smoke
    dg['swap_rates']=np.divide(result['swap_accepts'],result['swap_attempts'],out=np.zeros_like(result['swap_accepts'],float),where=result['swap_attempts']>0).tolist()
    dg['global_MH_acceptance']=result.get('global_accepts',0)/max(result.get('global_attempts',1),1)
    dg['fat_modes']=modes_1d(x[...,1]);dg['calibration_status']='synthetic_known' if context.calibration.metadata.get('synthetic') else 'measured_independent'
    dg['status']='synthetic_validated' if dg['converged'] and not smoke else 'sampling_not_converged'
    dg['multimodal_flag']=len(dg['fat_modes'])>1
    if dg['multimodal_flag'] and dg['converged'] and not smoke:dg['status']='posterior_multimodal'
    flat=x.reshape(-1,x.shape[-1]);median=np.median(flat,axis=0);res=posterior.likelihood.residual(median)
    dg['prediction_sse_at_marginal_medians']=float(res@res);dg['D']=posterior.likelihood.D
    from scipy.stats import chi2
    dg['nominal_covariance_residual_chi2_p']=float(chi2.sf(res@res,max(1,posterior.likelihood.D-x.shape[-1])))
    dg['nominal_noise_misfit_flag']=dg['nominal_covariance_residual_chi2_p']<.01
    dg['noise_misfit_note']='reference covariance with s2=1, descriptive; fitted s2 may absorb structural error'
    if dg['multimodal_flag']:
        indicators=[]
        for mode in dg['fat_modes']:
            if 'range' in mode:
                lo,hi=mode['range'];indicators.append(((x[...,1]>=lo)&(x[...,1]<=hi)).astype(float))
        if indicators:dg['mode_occupancy_diagnostics']=chain_diagnostics(np.stack(indicators,axis=-1))
    lo,hi=context.bounds[1];margin=.001*(hi-lo)
    dg['boundary_fraction']=float(np.mean((flat[:,1]<lo+margin)|(flat[:,1]>hi-margin)))
    # Posterior predictive discrepancy in the same real whitened coordinates.
    rng=np.random.default_rng(510)
    selected=np.linspace(0,len(flat)-1,100,dtype=int);scales=result['s2'].reshape(-1)[selected]
    discrepancy=np.array([posterior.likelihood.sse(flat[i])/s for i,s in zip(selected,scales)])
    dg['posterior_predictive_discrepancy_p']=float(np.mean(rng.chisquare(posterior.likelihood.D,len(selected))>=discrepancy))
    np.savez(run/'whitened_residuals.npz',residual_at_marginal_medians=res,posterior_discrepancy=discrepancy)
    json_write(run/'diagnostics.json',dg)
    summaries=[]
    for i,name in enumerate(names):
        summaries.append({'parameter':name,**interval_summary(x[...,i]),'map':float(map_candidates[0]['theta'][i]) if map_candidates else '',
                          'rhat':dg['rhat'][i],'ess_bulk':dg['ess_bulk'][i],'ess_tail':dg['ess_tail'][i],'mcse_mean':dg['mcse_mean'][i]})
    derived=2*(x[...,0]+x[...,1]);dd=chain_diagnostics(derived[...,None])
    summaries.append({'parameter':'d_cal_mm',**interval_summary(derived),'rhat':dd['rhat'][0],
                      'ess_bulk':dd['ess_bulk'][0],'ess_tail':dd['ess_tail'][0],'mcse_mean':dd['mcse_mean'][0]})
    summaries.append({'parameter':'s2',**interval_summary(result['s2']),
                      'map':map_candidates[0].get('s2','') if map_candidates else '',
                      'rhat':dg['rhat'][-1],'ess_bulk':dg['ess_bulk'][-1],
                      'ess_tail':dg['ess_tail'][-1],'mcse_mean':dg['mcse_mean'][-1]})
    with (run/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['parameter','mean','median','q025','q975','map','rhat','ess_bulk','ess_tail','mcse_mean']);writer.writeheader();writer.writerows(summaries)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(names),2,figsize=(10,2.3*len(names)),squeeze=False)
    for i,name in enumerate(names):
        axs[i,0].plot(x[:,:,i].T,alpha=.6,lw=.5);axs[i,0].set_ylabel(name)
        axs[i,1].hist(flat[:,i],bins=45,density=True)
    fig.tight_layout();fig.savefig(run/'posterior_trace.png',dpi=140);plt.close(fig)
    if len(names)>=4:
        fig,ax=plt.subplots();ax.scatter(flat[::max(1,len(flat)//3000),1],flat[::max(1,len(flat)//3000),2],s=2,alpha=.3)
        ax.set(xlabel='d_fat_mm',ylabel='u_eps_fat');fig.savefig(run/'fat_material_joint.png',dpi=140);plt.close(fig)
    from .likelihood import predict_observation
    from .operators import project_observation
    y=np.array([predict_observation(t,context) for t in flat[np.linspace(0,len(flat)-1,100,dtype=int)]])
    observed=context.observed.reshape(-1,14);op=context.operator
    z=observed[:,:7]+1j*observed[:,7:];yo=z@op.Q.T
    fig,axs=plt.subplots(2,2,figsize=(10,6));freq=op.f_full_hz[op.mask]/1e9
    transforms=[np.real,np.imag,np.abs,lambda v:np.unwrap(np.angle(v),axis=-1)]
    for ax,fn,label in zip(axs.ravel(),transforms,['real','imaginary','amplitude','phase display only']):
        a=fn(y);ax.fill_between(freq,np.quantile(a,.025,axis=0),np.quantile(a,.975,axis=0),alpha=.3)
        ax.plot(freq,fn(yo).T,alpha=.5);ax.set(xlabel='GHz',ylabel=label)
    fig.tight_layout();fig.savefig(run/'posterior_predictive.png',dpi=140);plt.close(fig)
    data_note='Synthetic experiment only.' if context.calibration.metadata.get('synthetic') else 'Measured data; estimate conditional on independent calibration and model.'
    (run/'report.md').write_text('Status: '+dg['status']+'\n\n'+data_note+' No human accuracy claim. Cold chains only.\n'
        +'Rank Rhat / bulk ESS / tail ESS include s2. Smoke never counts as converged validation.\n'
        +'Predictive bands show conditional mean response; measurement noise and unmodeled errors require separate checks.\n',encoding='utf-8')
    return dg
