from matplotlib import pyplot as plt
import numpy as np
import ephem

from compass.compass_network import Network
from compoundeye import fibonacci_sphere
from environment import Sky
from small_sensor.utils import smallest_diff2angles
from small_sensor.sensor import SimpleCompass
from sphere import azidist


class Trial(object):
    def __init__(self, sensor_class=None, sky_class=None, network_class=None,
                # experiment parameters
                samples=3
                 ):

        if sensor_class is None:
            self.sensor = SimpleCompass()
        else:
            self.sensor = sensor_class

        if sky_class is None:
            self.sky = Sky()
        else:
            self.sky = sky_class

        if network_class is None:
            self.network = Network(nb_pol=self.sensor.op_units)
        else:
            self.network = network_class

        self.samples = samples

    def generate_random_sun_positions(self):

        # generate random sun positions
        print('generating sun positions with fibonacci sphere')
        theta_s, phi_s = fibonacci_sphere(samples=self.samples, fov=161)
        sun_phis = phi_s[theta_s <= np.pi / 2]
        sun_thetas = theta_s[theta_s <= np.pi / 2]
        sun_thetas = (sun_thetas - np.pi) % (2 * np.pi) - np.pi
        sun_phis = (sun_phis + np.pi) % (2 * np.pi) - np.pi
        self.samples = sun_thetas.size
        return sun_thetas, sun_phis

    def run(self, sun_thetas=None, sun_phis=None, theta_tilt=0.3, phi_tilt=0.):

        self.sensor.phi_t = phi_tilt
        self.sensor.theta_t = theta_tilt
        if sun_thetas and sun_phis:
            # todo - sort out for different input types - list, float ndarray
            # sun_thetas = np.array(sun_thetas)
            # sun_phis = np.array(sun_phis)
            # print('sun angles: ', sun_thetas, sun_phis)
            # print('sun angles: ', type(sun_thetas), type(sun_phis))
            # print('sun angles: ', list(sun_thetas), list(sun_phis))
            pass
        else:
            sun_thetas, sun_phis = self.generate_random_sun_positions()
        self.samples = np.size(sun_thetas)

        # initialise results data containers
        azimuth_error_rad = np.zeros(self.samples, dtype=np.float32)
        # t = np.zeros_like(azimuth_error_rad)
        # d_eff = np.zeros((self.samples), dtype=np.float32)
        # a_ret = np.zeros_like(t)
        # tb1 = np.zeros((self.samples, self.network.nb_tcl), dtype=np.float32)

        for idx, (this_sun_theta, this_sun_phi) in enumerate(zip(sun_thetas, sun_phis)):
            # print('test params: theta {} phi {}'.format(this_sun_theta, this_sun_phi))
            # self.sky.sun_ele = this_sun_theta
            # self.sky.sun_azi = this_sun_phi
            self.sky.theta_s = this_sun_theta
            self.sky.phi_s = this_sun_phi
            # print ('sensor thetas: {}'.format(self.sensor.theta))
            # print ('sensor phis: {}'.format(self.sensor.phi))
            # luminance, dop, aop = self.sky(theta=self.sensor.theta,
            #                                phi=self.sensor.phi)
            luminance, dop, aop = self.sky(theta=self.sensor.theta, phi=self.sensor.phi,
                                           theta_tilt=theta_tilt, phi_tilt=phi_tilt,
                                           noise=0., eta=None,
                                           uniform_polariser=False)
            # luminance, dop, aop = self.sky.lum_aop_dop_from_position(theta_sensor=self.sensor.theta, phi_sensor=self.sensor.phi, noise=0., eta=None, uniform_polariser=False)
            # print('lum: {} dop: {} aop: {}'.format(luminance, dop, aop))
            self.network.compute(sensor=self.sensor, luminance=luminance, aop=aop, dop=dop)
            self.network.decode_response()

            # todo - work through this - just do azimuth error until others needed
            print('sun azimuth: {}, estimated azimuth {}, delta = {} degrees'.format(this_sun_phi, self.network.azimuth_pred,
                   np.rad2deg(smallest_diff2angles(this_sun_phi, self.network.azimuth_pred))))   # todo - seems to be always outputting the same value?
            azimuth_error_rad[idx] = np.absolute(azidist(np.array([this_sun_theta, this_sun_phi]), np.array([0., self.network.azimuth_pred])))
            # t[i, j] = tau_pred
            # a_ret[i, j] = a_pred
            # tb1[i, j] = r_tcl
            #
            # # effective degree of polarisation
            # M = r_sol.max() - r_sol.min()
            # # M = t[i, j] * 2.
            # p = np.power(10, M / 2.)
            # d_eff[i, j] = np.mean((p - 1.) / (p + 1.))

        d_deg = np.rad2deg(azimuth_error_rad)

            # d = azimuth error between sun position and estimated sun position in radians
            # d_deg = d in degrees
            # d_eff = effective degree of pol
            # t = tau = confidence in prediction
            # a_ret = a_pred = azimuth prediction
            # r_tcl = response of tcl neurons
        return d_deg #, d_eff, t, a_ret, tb1



def get_edinburgh_sky(date_str='2021/06/06 12:00'):

    # agent location
    edinburgh = ephem.city('Edinburgh')
    edinburgh.date = date_str   # todo - default to now?
    print('Edinburgh date ', edinburgh.date)

    # sun location
    sun = ephem.Sun()
    sun.date = date_str
    sun.compute(edinburgh)   # compute sun position relative to observer

    # lon_s, lat_s = sun2lonlat(sun, lonlat=True, show=True)
    # theta_s, phi_s = lonlat2theta_phi(lon_s, lat_s)
    sun_azi, sun_ele = sun2azi_ele(sun)
    print('sun azi: {} sun ele: {}'.format(sun_azi, sun_ele))
    sky = Sky(theta_s=sun_ele, phi_s=sun_azi)

    return sky, sun

def sun2azi_ele(s):
    """
    Returns the position of the sun relative to the obeserver it has been computed against
    s properties:
    s.az — Azimuth east of north
    s.alt — Altitude above horizon
    """
    azi = s.az.real
    ele = (np.pi/2) - s.alt.real  # todo - figure out why the coordinates start at zenith

    assert s.az.imag == 0, 'unknown behaviour imag az: {}'.format(s.az.imag)
    assert s.alt.imag == 0, 'unknown behaviour imag alt: {}'.format(s.alt.imag)
    # print('Sun azi: ' + str(azi) + ' ele: ' + str(ele))
    # print(azi, ele)
    return azi.real, ele.real

def trial_once():
    date_str = '2021/06/01 12:00'
    sky, sun = get_edinburgh_sky(date_str=date_str)
    sun_azi, sun_ele = sun2azi_ele(sun)

    #################### single trial, 8 pol sensors
    t = Trial()
    # err_deg = t.run(sun_thetas=[np.pi/4,], sun_phis=[np.pi/1.5,])
    test1_err_deg = t.run(sun_thetas=[sun_ele,], sun_phis=[sun_azi,])
    print(test1_err_deg)
    # visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=t.sensor)
    # fig, ax = t.network.plot_weight_states()
    # plt.show()

def trial_across_day():
    #################### day trial, 8 pol sensors
    t = Trial()
    # err_deg = t.run(sun_thetas=[np.pi/4,], sun_phis=[np.pi/1.5,])

    sun_azis = []
    sun_eles = []
    results = []
    hours = []
    for idx in range(24):
        hours.append(idx)
        date_str = '2021/06/01 ' + str(idx) + ':00'
        print (date_str)
        sky, sun = get_edinburgh_sky(date_str=date_str)
        sun_azi, sun_ele = sun2azi_ele(sun)

        res = t.run(sun_thetas=[sun_ele,], sun_phis=[sun_azi,])
        results.append(res)
        sun_azis.append(sun_azi)
        sun_eles.append(sun_ele)

    fig, ax = plt.subplots(3, 1) #, figsize=(12, 16))

    cax = ax[0]
    p = cax.scatter(hours, sun_eles)
    cax.title.set_text('Sun elevation')

    cax = ax[1]
    p = ax[1].scatter(hours, sun_azis)
    cax.title.set_text('Sun azimuth')

    cax = ax[2]
    p = ax[2].scatter(hours, results)
    cax.title.set_text('Compass Error')

    fig.suptitle("Edinburgh Sun passage on {}".format(date_str[0:10]), fontsize=14)
    plt.show()

if __name__ == '__main__':

    # trial_once()
    trial_across_day()
