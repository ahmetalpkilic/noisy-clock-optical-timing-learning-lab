"""Optical delay inference after errors-in-variables clock calibration."""
from pathlib import Path
import json
import numpy as np
from optics import image, fit_center, SIGMA, PIXELS, READ_SD, ORIGIN, VELOCITY

RATE, OFFSET, DELAY, FRAME = 1.002, .030, .040, 2.0
SYNC = np.linspace(-1., 1., 41)
REFERENCE_SD = .020


def calibrate(master, reference, master_sd, reference_sd=REFERENCE_SD):
    """Positive-slope Deming fit; master_sd=0 is the OLS control.

    Profile latent true times out of the Gaussian likelihood. The resulting
    objective is sum((y-m*x-b)^2)/(sy^2+m^2*sx^2). Its stationary point is below.
    """
    x, y = np.asarray(master, float), np.asarray(reference, float)
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 3
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or not np.isfinite(master_sd) or master_sd < 0
            or not np.isfinite(reference_sd) or reference_sd <= 0):
        raise ValueError('Require >=3 finite pairs, sx>=0 and sy>0')
    xc, yc = x-x.mean(), y-y.mean()
    xx, yy, xy = xc@xc, yc@yc, xc@yc
    if xx <= 1e-15 or xy <= 1e-15:
        raise ValueError('Clock observations must span time with positive covariance')
    if master_sd == 0:
        slope = xy/xx
    else:
        ratio = (reference_sd/master_sd)**2
        d = yy-ratio*xx
        root = np.hypot(d, 2*np.sqrt(ratio)*xy)
        # Rationalize the root when subtraction would lose precision.
        slope = (d+root)/(2*xy) if d >= 0 else 2*ratio*xy/(root-d)
    return np.array([slope, y.mean()-slope*x.mean()])


def delay_from_center(center, calibration):
    slope, intercept = calibration
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError('Clock rate must be positive')
    reference_time = (center-ORIGIN)/VELOCITY
    return (reference_time-intercept)/slope-FRAME


def latent_times(x, y, slope, intercept, sx, sy):
    """Optimal nuisance times at a fixed fitted line, for sx>0."""
    return (x/sx**2+slope*(y-intercept)/sy**2)/(1/sx**2+slope**2/sy**2)


def demo(out='outputs', trials=1000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20261003)
    center = ORIGIN+VELOCITY*(RATE*(FRAME+DELAY)+OFFSET)
    profile = image(center)
    # Each Monte Carlo trial includes an actual pixel-noise draw and image fit.
    positions = np.array([fit_center(profile+rng.normal(0,READ_SD,len(PIXELS)))
                          for _ in range(trials)])
    rows=[]; saved=None
    for sx in [0., .03, .075, .15, .25]:
        estimates = {k:[] for k in ['ols','deming','half_master_sd']}
        slopes = {k:[] for k in estimates}
        for i in range(trials):
            x = SYNC+rng.normal(0,sx,len(SYNC))
            y = RATE*SYNC+OFFSET+rng.normal(0,REFERENCE_SD,len(SYNC))
            for name, assumed in [('ols',0.),('deming',sx),('half_master_sd',sx/2)]:
                fit=calibrate(x,y,assumed)
                estimates[name].append(delay_from_center(positions[i],fit))
                slopes[name].append(fit[0])
            if sx == .15 and i == 0: saved=(x,y)
        row={'master_sd_s':sx,
             'large_sample_ols_bias_s':(FRAME+DELAY)*sx**2/np.var(SYNC)}
        for name in estimates:
            error=np.array(estimates[name])-DELAY
            row[name]={'bias_s':float(error.mean()),'sd_s':float(error.std(ddof=1)),
                       'rmse_s':float(np.sqrt(np.mean(error**2))),
                       'mean_rate':float(np.mean(slopes[name]))}
        rows.append(row)
    report={'seed':20261003,'trials_per_level':trials,'sigma_px':SIGMA,
            'truth':{'rate':RATE,'offset_s':OFFSET,'delay_s':DELAY,'frame_s':FRAME},
            'center_sd_px':float(positions.std(ddof=1)),
            'noiseless_delay_s':float(delay_from_center(center,calibrate(SYNC,RATE*SYNC+OFFSET,.15))),
            'sensitivity':rows}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax[0].plot(PIXELS,profile-20,'o-'); ax[0].set(xlim=(-1,5),xlabel='Sensor coordinate (px)',ylabel='Signal (ADU)',title='Pixel-integrated diffraction spot')
    x,y=saved; ax[1].errorbar(x,y,xerr=.15,yerr=REFERENCE_SD,fmt='.',alpha=.45)
    grid=np.linspace(-1.4,2.1,100)
    ax[1].plot(grid,RATE*grid+OFFSET,'k--',label='True clock')
    for name,sx in [('OLS',0),('Deming',.15)]:
        m,b=calibrate(x,y,sx); ax[1].plot(grid,m*grid+b,label=name)
    ax[1].set(xlabel='Observed master time (s)',ylabel='Observed reference time (s)',title='Uncertainty in both timestamps'); ax[1].legend()
    for name,label in [('ols','OLS'),('deming','Correct Deming'),('half_master_sd','Understate master SD by 2x')]:
        ax[2].plot([r['master_sd_s']*1000 for r in rows], [r[name]['bias_s']*1000 for r in rows],'o-',label=label)
    ax[2].plot([r['master_sd_s']*1000 for r in rows],[r['large_sample_ols_bias_s']*1000 for r in rows],'k--',label='Large-sample OLS prediction')
    ax[2].set(xlabel='Master timestamp SD (ms)',ylabel='Mean delay error (ms)',title='Extrapolation turns rate bias into delay bias'); ax[2].legend(fontsize=7)
    for a in ax: a.grid(alpha=.2)
    fig.savefig(out/'experiment.png',dpi=160); plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__': demo()
