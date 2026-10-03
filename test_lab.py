import unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
import optics as o
import lab as l

class Tests(unittest.TestCase):
    def test_independent_pupil_fourier(self):
        rho=np.linspace(-8000,8000,65536,endpoint=False); step=rho[1]-rho[0]
        pupil=np.exp(-(rho/o.PUPIL_RADIUS_UM)**2)
        intensity=abs(np.fft.fftshift(np.fft.fft(pupil)))**2
        u=np.fft.fftshift(np.fft.fftfreq(len(rho),step))*o.WAVELENGTH_UM*o.FOCAL_UM/o.PITCH_UM
        sd=np.sqrt(np.sum(u*u*intensity)/intensity.sum())
        self.assertAlmostEqual(sd,o.SIGMA,places=12)

    def test_pixel_integration_flux_derivative(self):
        c=.27
        independent=np.array([quad(lambda x: np.exp(-.5*((x-c)/o.SIGMA)**2)/(o.SIGMA*np.sqrt(2*np.pi)),j-.5,j+.5,epsabs=1e-13)[0] for j in o.PIXELS])
        np.testing.assert_allclose(o.image(c)-o.BACKGROUND,o.FLUX*independent,atol=1e-10)
        self.assertAlmostEqual(sum(o.image(c)-o.BACKGROUND),o.FLUX,places=9)
        np.testing.assert_allclose(o.derivative(c),(o.image(c+1e-6)-o.image(c-1e-6))/2e-6,atol=2e-6)
        self.assertAlmostEqual(o.fit_center(o.image(c)),c,places=10)

    def test_known_answer_and_zero_axis_noise(self):
        y=l.RATE*l.SYNC+l.OFFSET
        for sx in [0,.15,1e-9]:
            np.testing.assert_allclose(l.calibrate(l.SYNC,y,sx),[l.RATE,l.OFFSET],atol=1e-13)
        center=o.ORIGIN+o.VELOCITY*(l.RATE*(l.FRAME+l.DELAY)+l.OFFSET)
        self.assertAlmostEqual(l.delay_from_center(center,l.calibrate(l.SYNC,y,.15)),l.DELAY,places=12)
        rng=np.random.default_rng(4); yn=y+rng.normal(0,.02,len(y))
        expected=np.linalg.lstsq(np.column_stack((l.SYNC,np.ones(len(y)))),yn,rcond=None)[0]
        np.testing.assert_allclose(l.calibrate(l.SYNC,yn,0),expected,atol=1e-14)

    def test_latent_gaussian_optimization(self):
        rng=np.random.default_rng(21); x=l.SYNC+rng.normal(0,.15,len(l.SYNC)); y=l.RATE*l.SYNC+l.OFFSET+rng.normal(0,.02,len(x))
        fit=l.calibrate(x,y,.15); latent=l.latent_times(x,y,*fit,.15,.02)
        def residual(z): return np.r_[(x-z[2:])/.15,(y-z[0]*z[2:]-z[1])/.02]
        numerical=least_squares(residual,np.r_[1.,0.,x],gtol=1e-11,ftol=1e-12,xtol=1e-12)
        np.testing.assert_allclose(numerical.x[:2],fit,atol=2e-8)
        np.testing.assert_allclose(numerical.x[2:],latent,atol=2e-8)

    def test_population_attenuation_and_correction(self):
        rng=np.random.default_rng(27); n=200000; t=rng.uniform(-1,1,n)
        x=t+rng.normal(0,.15,n); y=l.RATE*t+l.OFFSET+rng.normal(0,.02,n)
        ols=l.calibrate(x,y,0); deming=l.calibrate(x,y,.15)
        self.assertLess(abs(ols[0]-l.RATE/(1+3*.15**2)),.002)
        self.assertLess(abs(deming[0]-l.RATE),.002)
        center=o.ORIGIN+o.VELOCITY*(l.RATE*(l.FRAME+l.DELAY)+l.OFFSET)
        self.assertGreater(l.delay_from_center(center,ols)-l.DELAY,.12)
        self.assertLess(abs(l.delay_from_center(center,deming)-l.DELAY),.005)

    def test_shift_invariance(self):
        rng=np.random.default_rng(2); x=l.SYNC+rng.normal(0,.15,len(l.SYNC)); y=l.RATE*l.SYNC+l.OFFSET
        m,b=l.calibrate(x,y,.15); m2,b2=l.calibrate(x+4,y-2,.15)
        np.testing.assert_allclose([m2,b2],[m,b-2-4*m],atol=1e-12)

    def test_invalid_inputs(self):
        for x,y,sx,sy in [([1,2],[1,2],.1,.1),([1,1,1],[1,2,3],.1,.1),([1,2,3],[3,2,1],.1,.1),([1,2,3],[1,2,3],-.1,.1),([1,2,3],[1,2,3],.1,0),([1,2,3],[1,2,np.nan],.1,.1)]:
            with self.assertRaises(ValueError): l.calibrate(x,y,sx,sy)

if __name__=='__main__': unittest.main()
