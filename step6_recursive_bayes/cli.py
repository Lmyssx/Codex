import argparse,copy,json
from pathlib import Path
import numpy as np
from .io import load_config,resolve,scan_records,audit_records,load_raw_mat,raw_windows,file_hash
from .reporting import new_run,json_write,save_map,save_posterior
from .simulate import build_context,simulate_record,make_posterior
from .operators import apply_step5_operator,project_observation
from .calibration import load_calibration
from .likelihood import shrink_covariance,record_covariance
from .samplers import sample_pt

def estimate_noise(config,records,run):
    paths=sorted((resolve(config,config['data']['raw_root'])/config['data']['empty_subdir']).glob('*.mat'))
    if not paths:
        result={'status':'raw_empty_data_missing','covariance_estimated':False};json_write(run/'noise_status.json',result);return result
    r=records[0];background=np.mean([load_raw_mat(p).mean(axis=0) for p in paths],axis=0);points=[]
    for p in paths:
        hw=raw_windows(load_raw_mat(p),r,background)
        y=apply_step5_operator(np.fft.fftshift(np.fft.fft(hw,axis=-1),axes=-1),r.operator)
        points.extend(project_observation(y,r.operator.Q))
    C,details=shrink_covariance(points,config.get('noise',{}).get('shrinkage',.2))
    np.savez(run/'noise_covariance.npz',covariance=C,points=points,f_obs_hz=r.f_obs_hz,
             gate_bins=r.operator.gate_bins,Q=r.operator.Q,nfft=r.operator.nfft,dt_s=r.operator.dt_s)
    json_write(run/'noise_status.json',{'status':'estimated_requires_correlation_validation',**details,'inputs':[{'path':str(p),'sha256':file_hash(p)} for p in paths]})
    return C

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','validate-physics','estimate-noise','simulate','recover-synthetic','map','sample','report','validate-statistics','validate-coverage','validate-recovery'])
    parser.add_argument('--config');parser.add_argument('--record');parser.add_argument('--run-dir');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='report':
        p=Path(args.run_dir);print((p/'report.md').read_text(encoding='utf-8'));return
    c=load_config(args.config);run=Path(args.run_dir) if args.resume else new_run(c,args.command)
    if c['calibration'].get('allow_blind_waveform',False):raise ValueError('blind waveform inversion is outside this implementation')
    cmd=args.command
    if cmd in ['validate-physics','validate-statistics','validate-coverage','validate-recovery']:
        from .validation import physics_validation,statistics_validation,coverage_validation,recovery_validation
        result={'validate-physics':physics_validation,'validate-statistics':statistics_validation,'validate-coverage':coverage_validation,'validate-recovery':recovery_validation}[cmd](c,run)
        json_write(run/'validation.json',result)
    elif cmd in ['audit','estimate-noise','map','sample']:
        records=scan_records(resolve(c,c['data']['observations_root']))
        audit=audit_records(records,resolve(c,c['data']['raw_root']),c['data']['empty_subdir']);json_write(run/'audit.json',audit)
        if cmd=='audit':result={'status':'audit_only','record_count':len(records)}
        elif cmd=='estimate-noise':result=estimate_noise(c,records,run)
        else:
            matched=[r for r in records if r.path.stem==args.record or r.source_file==args.record]
            if len(matched)!=1:raise ValueError('record must uniquely match source_file or observation stem')
            r=matched[0]
            try:cal=load_calibration(resolve(c,c['calibration']['path']) if c['calibration']['path'] else None,r.operator,c['calibration'].get('target_metadata'))
            except FileNotFoundError:
                result={'status':'calibration_missing','record':r.source_file,'thickness_estimate':None};json_write(run/'diagnostics.json',result)
                (run/'report.md').write_text('calibration_missing: no real thickness inference performed.\n',encoding='utf-8')
                print(json.dumps({'run_dir':str(run),**result}));return
            if c['model']['g_source'] in ['unverified_placeholder',None]: raise ValueError('independent geometry required')
            selected=np.asarray(c['data'].get('observation_indices',list(range(len(r.H_obs)))))
            if (selected.ndim!=1 or len(selected)==0 or not np.issubdtype(selected.dtype,np.integer)
                or len(set(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=len(r.H_obs))):
                raise ValueError('invalid selected observation indices')
            if len(selected)>1 and not c.get('noise',{}).get('validated_repeat_structure',False):raise ValueError('repeat/covariance validation required before pooling windows')
            context=build_context(c,delay=cal.metadata.get('residual_delay_sd_s',0)>0);context.operator=r.operator;context.calibration=cal
            context.calibration_delay_sd_s=cal.metadata.get('residual_delay_sd_s',0.)
            if context.calibration_delay_sd_s and not cal.metadata.get('delay_covariance_source'):
                raise ValueError('independent repeat provenance required for calibration delay covariance')
            if context.calibration_delay_sd_s and cal.metadata.get('delay_basis')!='center_anchored_group_delay':
                raise ValueError('residual delay basis must match implemented center-anchored group delay')
            from .materials import load_material_model,cole_cole
            model=load_material_model(resolve(c,c['materials']['ascii_path']))
            context.baseline={k:cole_cole(r.operator.f_full_hz,p) for k,p in model.parameters.items()}
            from .likelihood import configure_long_convolution
            configure_long_convolution(context,model,c['model'].get('linear_convolution_factor',8))
            context.observed=project_observation(r.H_obs[selected,0],r.operator.Q)
            scatter=np.linalg.norm(context.observed-context.observed.mean(axis=0))/max(np.linalg.norm(context.observed),1e-300)
            if len(selected)>1 and scatter>.2:
                raise ValueError('window drift requires an explicit geometry/drift model; current implementation supports selecting one predetermined window')
            if not c.get('noise',{}).get('path'):raise ValueError('noise_covariance_missing')
            with np.load(resolve(c,c['noise']['path']),allow_pickle=False) as z:
                C=z['covariance']
                for key in ['f_obs_hz','gate_bins','Q','nfft','dt_s']:
                    if key not in z:raise ValueError('noise operator provenance missing: '+key)
                if (not np.allclose(z['f_obs_hz'],r.f_obs_hz,rtol=1e-14,atol=1e-6)
                    or not np.array_equal(z['gate_bins'],r.operator.gate_bins)
                    or not np.allclose(z['Q'],r.operator.Q,atol=1e-13)
                    or int(z['nfft'])!=r.operator.nfft or not np.isclose(float(z['dt_s']),r.operator.dt_s,rtol=1e-14,atol=0)):
                    raise ValueError('noise operator provenance mismatch')
            context.covariance=record_covariance(C,len(selected),c['noise'].get('rho',0),c['noise'].get('shared_fraction',0),selected)
            context.g_air_m=c['model']['g_air_mm']*1e-3
            result=run_inference(context,c,run,cmd=='sample',args.resume)
            for r in records:
                if file_hash(r.path)!=r.sha256:raise RuntimeError('original observations modified')
    else:
        context=build_context(c);truth=np.array(c['synthetic']['truth'])
        y=simulate_record(context,truth,c['seed'],c['synthetic'].get('noiseless',False))
        np.savez(run/'synthetic_data.npz',observed=context.observed,truth=truth,mean_spectrum=y,
                 system_response_full=context.calibration.system_response_full,f_full_hz=context.operator.f_full_hz,
                 covariance=context.covariance)
        result={'status':'synthetic_generated','not_measured_calibration':True} if cmd=='simulate' else run_inference(context,c,run,True,args.resume)
    if isinstance(result,dict):json_write(run/'command_status.json',result)
    if not (run/'report.md').exists():
        status=result.get('status','completed') if isinstance(result,dict) else 'completed'
        (run/'report.md').write_text('Command: '+cmd+'\n\nStatus: '+status+'\n\n'
            +'See command_status.json / audit.json / validation.json for measured evidence.\n'
            +'No real human thickness or accuracy claim. Missing raw MAT, calibration and labels are tracked separately.\n',encoding='utf-8')
    print(json.dumps({'run_dir':str(run),'status':result.get('status','completed') if isinstance(result,dict) else 'completed'}))

def run_inference(context,c,run,sample,resume=False):
    post=make_posterior(context);candidates,grid=__import__('step6_recursive_bayes.inference',fromlist=['run_map']).run_map(post,tuple(c.get('grid_shape',[19,80])))
    save_map(run,candidates,grid)
    if not sample:return {'status':'calibrated_conditional_estimate','accuracy_validated':False}
    settings=copy.deepcopy(c['sampler']);E=settings['independent_ensembles']
    if settings.get('use_map_mixture',False):
        from .inference import map_mixture_proposal
        settings['global_proposal']=map_mixture_proposal(post,candidates)
        json_write(run/'frozen_global_proposal.json',settings['global_proposal'])
    target_bytes=context.observed.tobytes()+context.covariance.tobytes()+context.calibration.system_response_full.tobytes()+context.calibration.residual_offset_full.tobytes()+context.operator.f_full_hz.tobytes()
    for value in context.baseline.values():target_bytes+=value.tobytes()
    settings['target_id']=file_hash(run/'config_resolved.json')+__import__('hashlib').sha256(target_bytes).hexdigest()
    rng=np.random.default_rng(c['seed']+1)
    starts=[]
    for i in range(E):
        t=candidates[i%len(candidates)]['theta'].copy()
        if i>0:t[:2]=np.clip(t[:2]+rng.normal(size=2)*[.1,1],context.bounds[:2,0],context.bounds[:2,1])
        starts.append(t)
    result=sample_pt(post,starts,settings,c['seed'],run/'checkpoint.npz',resume=resume)
    dg=save_posterior(run,result,context,post,candidates,c.get('smoke',False))
    if not context.calibration.metadata.get('synthetic') and dg['status']=='synthetic_validated':
        dg['status']='calibrated_conditional_estimate';json_write(run/'diagnostics.json',dg)
    return dg

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,FloatingPointError) as exc:
        import sys
        print(json.dumps({'status':'input_or_model_error','error':str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
