from matplotlib import pyplot as plt
import numpy as np

from compass.compass_network import Network

from environment.sky import sun2azi_ele, get_edinburgh_sky, Sky
from small_sensor.sensor import SimpleCompass, SphericalDistributionCompass
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

        self.FFT_r_tcl = np.nan       # response of tcl neurons using fast fourier transform
        self.azimuth_pred = np.nan    # azimuth prediction
        self.tau_pred = np.nan        # confidence in prediction

    def run(self, sun_theta=None, sun_phi=None, theta_tilt=0., phi_tilt=0.):

        # tilt the sensor
        self.sensor.tilt_sensor(theta_tilt=theta_tilt, phi_tilt=phi_tilt)

        # set sun position
        self.sky.theta_s = sun_theta
        self.sky.phi_s = sun_phi

        # read the sky information with each pol unit
        luminance, dop, aop = self.sky(sensor=self.sensor,
                                       theta_tilt=theta_tilt, phi_tilt=phi_tilt,
                                       noise=0., eta=None,
                                       uniform_polariser=False)

        # print('lum: {} dop: {} aop: {}'.format(luminance, dop, aop))

        # compute the response using the sensor network
        self.network.compute(sensor=self.sensor, luminance=luminance, aop=aop, dop=dop)
        self.FFT_r_tcl, self.azimuth_pred, self.tau_pred = self.network.decode_response()

        # print('sun azimuth: {}, estimated azimuth {}, delta = {} degrees'.format(sun_phi, self.network.azimuth_pred,
        #        np.rad2deg(smallest_diff2angles(sun_phi, self.network.azimuth_pred))))

        # todo - should we use the tilted sun azimuth as the reference?
        # this_sun_theta_tilted = self.sky.theta_sun_tilted
        # this_sun_phi_tilted = self.sky.phi_sun_tilted
        # azimuth_error_rad = np.absolute(azidist(np.array([this_sun_theta_tilted, this_sun_phi_tilted]), np.array([0., self.network.azimuth_pred])))
        azimuth_error_rad = np.absolute(azidist(np.array([sun_theta, sun_phi]), np.array([0., self.network.azimuth_pred])))

        d_deg = np.rad2deg(azimuth_error_rad)

        return d_deg


def trial_once():
    date_str = '2021/06/01 12:00'
    sky, sun = get_edinburgh_sky(date_str=date_str)
    sun_azi, sun_ele = sun2azi_ele(sun)

    #################### single trial, 8 pol sensors
    t = Trial()
    # err_deg = t.run(sun_thetas=[np.pi/4,], sun_phis=[np.pi/1.5,])
    test1_err_deg = t.run(sun_theta=[sun_ele, ], sun_phi=[sun_azi, ])
    print(test1_err_deg)
    # visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=t.sensor)
    # fig, ax = t.network.plot_weight_states()
    # plt.show()

def trial_across_day(sensor=None):
    #################### day trial, 8 pol sensors
    # sensor = FibonacciCompass()
    if not sensor:
        sensor = SimpleCompass()
    # t = Trial(sensor_class=sensor)
    t = Trial(sensor_class=sensor)

    sun_azis = []
    sun_eles = []
    sensor_azis = []
    results = []
    hours = []
    for idx in range(24):
        hours.append(idx)
        date_str = '2021/06/01 ' + str(idx) + ':00'
        # print(date_str)
        sky = get_edinburgh_sky(date_str=date_str)
        res = t.run(sun_theta=sky.theta_s, sun_phi=sky.phi_s)
        results.append(res)
        sensor_azis.append((t.azimuth_pred) % (2*np.pi))
        sun_eles.append(sky.theta_s)
        sun_azis.append(sky.phi_s)
        print((t.azimuth_pred) )
        print(len(sun_eles), len(sun_azis), len(sensor_azis))

    fig, ax = plt.subplots(2, 1) #, figsize=(12, 16))
    cax = ax[0]
    p1 = cax.scatter(sun_azis, sun_eles, marker="^", label='actual')
    # print(len(sun_eles), len(sun_azis), len(sensor_azis))
    # print(sensor_azis)
    p2 = cax.scatter(sensor_azis, sun_eles, label='estimated')
    p1.set_alpha(0.5)
    p2.set_alpha(0.5)
    cax.title.set_text('Sun ephemeris function')
    cax.set_ylabel('sun elevation $(\circ)$')
    cax.set_xlabel('sun azimuth $(\circ)$')
    cax.grid()
    cax.legend()

    cax = ax[1]
    p = ax[1].scatter(hours, results)
    cax.title.set_text('Compass Error')
    cax.set_xlabel('Time (hour)')
    cax.set_ylabel('Azimuth error $(\circ)$')
    inc = 4
    cax.set_xticks(np.arange(0, 24 + inc, inc))
    cax.grid()

    fig.suptitle("Edinburgh Sun passage on {}".format(date_str[0:10]), fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # fig, ax = plt.subplots(3, 1) #, figsize=(12, 16))
    # cax = ax[0]
    # p = cax.scatter(hours, sun_eles)
    # cax.title.set_text('Sun elevation')
    # cax = ax[1]
    # p = ax[1].scatter(hours, sun_azis)
    # cax.title.set_text('Sun azimuth')
    # cax = ax[2]
    # p = ax[2].scatter(hours, results)
    # cax.title.set_text('Compass Error')
    # fig.suptitle("Edinburgh Sun passage on {}".format(date_str[0:10]), fontsize=14)
    # plt.show()

if __name__ == '__main__':

    # trial_once()
    sensor = SphericalDistributionCompass()
    # sensor = SimpleCompass()
    trial_across_day(sensor=sensor)
