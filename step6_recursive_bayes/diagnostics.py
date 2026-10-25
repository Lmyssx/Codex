import numpy as np
from scipy.signal import find_peaks

def chain_diagnostics(samples):
    import arviz_stats as az
    x=np.asarray(samples,float)
    if x.ndim!=3 or x.shape[0]<2 or x.shape[1]<20:raise ValueError('multiple independent cold chains required')
    rh=np.asarray(az.rhat(x,chain_axis=0,draw_axis=1,method='rank'))
    bulk=np.asarray(az.ess(x,chain_axis=0,draw_axis=1,method='bulk'))
    tail=np.asarray(az.ess(x,chain_axis=0,draw_axis=1,method='tail',prob=(.05,.95)))
    mcse=np.asarray(az.mcse(x,chain_axis=0,draw_axis=1,method='mean'))
    converged=bool(np.all(rh<1.01) and np.all(bulk>=400) and np.all(tail>=400))
    return {'rhat':rh.tolist(),'ess_bulk':bulk.tolist(),'ess_tail':tail.tolist(),'mcse_mean':mcse.tolist(),
            'converged':converged,'criteria':'rank Rhat<1.01 and bulk/tail ESS>=400 for every parameter including s2'}

def modes_1d(x,bins=50):
    values=np.sort(np.asarray(x).ravel())
    cuts=np.flatnonzero(np.diff(values)>.5)+1
    clusters=np.split(values,cuts)
    supported=[v for v in clusters if len(v)/len(values)>=.005]
    if len(supported)>1:
        return [{'center':float(np.median(v)),'estimated_basin_probability':float(len(v)/len(values)),
                 'range':[float(v.min()),float(v.max())],'method':'separated sample clusters; gap>0.5 mm and mass>=0.5%'} for v in supported]
    h,edges=np.histogram(values,bins=bins,density=False)
    from scipy.ndimage import gaussian_filter1d
    smooth=gaussian_filter1d(h.astype(float),2.)
    peaks,_=find_peaks(smooth,prominence=max(smooth.max()*.15,1),distance=4)
    if len(peaks)==0:peaks=np.array([int(np.argmax(smooth))])
    cuts=[0]+[int(a+np.argmin(smooth[a:b+1])) for a,b in zip(peaks[:-1],peaks[1:])]+[len(h)]
    return [{'center':float((edges[i]+edges[i+1])/2),'estimated_basin_probability':float(h[cuts[j]:cuts[j+1]].sum()/max(h.sum(),1)),
             'method':'smoothed histogram; exploratory mode basins'} for j,i in enumerate(peaks)]

def interval_summary(x):
    x=np.asarray(x);return {'mean':float(x.mean()),'median':float(np.median(x)),
                             'q025':float(np.quantile(x,.025)),'q975':float(np.quantile(x,.975))}
