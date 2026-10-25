from pathlib import Path
import importlib.util
import numpy as np
import pytest
from step6_recursive_bayes.operators import *
from step6_recursive_bayes.reflection import *
from step6_recursive_bayes.materials import *
from step6_recursive_bayes.io import *
ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'Thermal_dielectric_acoustic_MR properties_database_V5.0(ASCII).txt'

def test_all_observations_readonly():
    records=scan_records(ROOT/'Step5_observations/observations');audit=audit_records(records)
    assert audit['record_count']==40 and audit['window_count']==400
    assert max(r['subspace_error'] for r in audit['entries'])<1e-12
    assert all(file_hash(r.path)==r.sha256 for r in records)

@pytest.mark.parametrize('layers',[2,3,4,6])
def test_random_tmm(layers):
    rng=np.random.default_rng(layers);f=make_operator().f_full_hz
    for _ in range(12):
        eps=rng.uniform(2,45,(layers,len(f)))-1j*rng.uniform(.01,8,(layers,len(f)));eps[0]=1
        d=rng.uniform(0,.02,layers-2)
        a=recursive_reflection(f,eps,d,.001);b=consistent_tmm(f,eps,d,.001)
        assert np.allclose(a,b,atol=1e-10,rtol=1e-10)
        assert np.max(abs(a))<=1+1e-12

def test_physical_limits():
    f=make_operator().f_full_hz;eps=np.array([np.ones(len(f)),np.full(len(f),4-2j),np.full(len(f),9-4j)])
    assert np.max(abs(recursive_reflection(f,np.ones((4,len(f))),[.01,.02])))<1e-14
    direct=(1/np.sqrt(eps[-1])-1)/(1/np.sqrt(eps[-1])+1)
    assert np.allclose(recursive_reflection(f,eps,[0]),direct)
    top=(1/np.sqrt(eps[1])-1)/(1/np.sqrt(eps[1])+1)
    assert np.allclose(recursive_reflection(f,eps,[10]),top)
    k,_=wave_parameters(f,eps);assert np.max(abs(np.exp(-1j*k*.02)))<=1+1e-14
    assert np.all(recursive_reflection(f,np.array([np.ones(len(f)),np.full(len(f),4)]),[]).real<0)

def test_grid_materials_match_existing_csv():
    model=load_material_model(DB);rows=np.genfromtxt(ROOT/'step6_tmm_forward_nopsy/tissue_properties_itis_v5_fobs.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    for name,p in model.parameters.items():
        r=rows[rows['tissue']==name];f=r['frequency_hz'];expected=r['eps_r']-1j*r['sigma_s_per_m']/(2*np.pi*f*8.8541878128e-12)
        assert np.allclose(cole_cole(f,p),expected,rtol=2e-14,atol=2e-14)
        assert np.all(np.isfinite(cole_cole(make_operator().f_full_hz,p)))
    assert model.units_raw['tau3']=='µs'

def test_step5_original_function_and_projection():
    spec=importlib.util.spec_from_file_location('original',ROOT/'Step5_observations/build_observations.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rng=np.random.default_rng(4);op=make_operator();h=rng.normal(size=96)+1j*rng.normal(size=96);hg=np.zeros_like(h);hg[op.gate_bins]=h[op.gate_bins]
    f,y=old.frequency_domain_response(hg,Nfft=96)
    new=apply_step5_operator(np.fft.fftshift(np.fft.fft(h)),op)
    assert np.array_equal(f,op.f_full_hz[op.mask]);assert np.allclose(y,new,atol=1e-14)
    assert op.Q.shape==(49,7) and subspace_error(y,op.Q)<1e-12
    assert project_observation(y,op.Q).shape==(14,)
    wrong=np.fft.fftshift(np.fft.fft(h[op.gate_bins],n=96))[op.mask]
    assert not np.allclose(y,wrong)

def test_gate_product_noncommutes():
    rng=np.random.default_rng(8);op=make_operator();s=rng.normal(size=96)+1j*rng.normal(size=96);x=np.exp(-1j*np.arange(96)*.1)
    assert np.linalg.norm(apply_step5_operator(s*x,op)-apply_step5_operator(s,op)*x[op.mask])>1

def test_fft_delay_phase_slope():
    op=make_operator();tau=3*op.dt_s;full=np.exp(-2j*np.pi*(op.f_full_hz-op.fc_hz)*tau)
    y=apply_step5_operator(full,op);slope=np.polyfit(op.f_full_hz[op.mask],np.unwrap(np.angle(y)),1)[0]
    assert np.isclose(slope,-2*np.pi*tau,rtol=1e-12)

def test_synthetic_mat_step5_reconstruction(tmp_path):
    from scipy.io import savemat
    spec=importlib.util.spec_from_file_location('old',ROOT/'Step5_observations/build_observations.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rng=np.random.default_rng(51);raw=rng.normal(size=(1100,96))+1j*rng.normal(size=(1100,96));bg=np.full(96,.2+.1j)
    p=tmp_path/'synthetic.mat';savemat(p,{'data':raw});result=old.process_body_file(p,bg,True)
    hw=(raw-bg).reshape(10,110,96).mean(axis=1)
    y=apply_step5_operator(np.fft.fftshift(np.fft.fft(hw,axis=-1),axes=-1),make_operator())[:,None]
    assert np.allclose(result['H_obs'],y,rtol=1e-12,atol=1e-14)
    # This tests code agreement on generated MAT, not original MAT reproduction.

@pytest.mark.parametrize('count',[3,6])
def test_zero_thickness_telescopes(count):
    op=make_operator();eps=np.array([np.full(96,2+i-1j*i) for i in range(count)]);eps[0]=1
    eta=1/np.sqrt(eps[-1]);expected=(eta-1)/(eta+1)
    assert np.allclose(recursive_reflection(op.f_full_hz,eps,np.zeros(count-2)),expected,atol=1e-13)

def test_unknown_material_unit_rejected(tmp_path):
    target=tmp_path/'bad.txt';target.write_bytes(DB.read_bytes().replace(b'tau3 (\xb5s)',b'tau3 (xs)'))
    with pytest.raises(ValueError,match='time unit'):load_material_model(target)

def test_frequency_mismatch_rejected(tmp_path):
    source=next((ROOT/'Step5_observations/observations').rglob('*_observations.npz'))
    with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    data['f_obs']=data['f_obs']+1e6;p=tmp_path/'invalid_observations.npz';np.savez(p,**data)
    with pytest.raises(ValueError,match='frequency_metadata_mismatch'):load_observation_record(p)
