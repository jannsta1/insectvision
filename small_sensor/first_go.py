import matplotlib.pyplot as plt
from datetime import datetime
import healpy as hp
import numpy as np
import ephem

# from compoundeye.evaluation import evaluate
from compoundeye.model import CompoundEye, DRA
from environment.sky import visualise_lum_angle_degree
from compoundeye.geometry import fibonacci_sphere
from environment import Sky
from compoundeye.model import visualise
from compass.compass_network import Network


class SimpleCompass(DRA):
    def __init__(self, op_units=8, ele=np.deg2rad(45), omega=56, rho=5.4):

        self.op_units = op_units   # todo - feed this notation down
        azis = np.linspace(0, 2 * np.pi, self.op_units, endpoint=False)
        ele = (np.pi/2) - np.tile(ele, self.op_units)  # todo - why does elevation start at zenith?
        # self.alpha =
        super().__init__(thetas=ele, phis=azis, n=self.op_units, omega=omega, rho=rho)

op_units = 8
unit_fov = np.deg2rad(10)
ele = np.deg2rad(45)
compass = SimpleCompass(op_units=op_units, ele=ele, omega=unit_fov, rho=unit_fov)


def sun2lonlat(s, lonlat=False, show=False):
    """
    Returns the position of the sun relative to the obeserver it has been computed against

    s properties:
    s.az — Azimuth east of north
    s.alt — Altitude above horizon
    """
    lon, lat = s.az, s.alt
    colat = (np.pi/2) - lat  # JS - have updated from: np.pi / 2 - lat
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    if show:
        print('Sun:\tLon = %.2f\t Lat = %.2f\t Co-Lat = %.2f' % \
              (np.rad2deg(lon), np.rad2deg(lat), np.rad2deg(colat)))

    if lonlat:  # return the longitude and the latitude in degrees
        return np.rad2deg(lon), np.rad2deg(lat)
    else:  # return the longitude and the co-latitude in radians
        return lon, colat

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

# def lonlat2theta_phi(lon, lat, theta_z=0, phi_z=0, alt=0):
#     # calculate the angular distance between the sun and every point on the map
#     x, y, z = 0, np.rad2deg(lat), -np.rad2deg(lon)
#     theta_s, phi_s = hp.Rotator(rot=(z, y, x))(theta_z, phi_z)  # todo - remove hp rotator
#     theta_s, phi_s = theta_s % np.pi, (phi_s + np.pi) % (2 * np.pi) - np.pi
#     return theta_s, phi_s


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


def smallest_diff2angles(angle_a, angle_b, rads=True):
    a = angle_a - angle_b
    if rads:
        return (a + np.pi) % (2*np.pi) - np.pi
    else:
        return (a + 180) % 360 - 180


def visualise_luminance_2(sky, sun=None, sensor=None, show=True, ax=None):

    fig = plt.figure()
    if not ax:
        plt.figure("Luminance", figsize=(4.5, 4.5))
        # ax = plt.subplot(111, polar=True)
        ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    samples = 1000
    fov = 180
    thetas, phis = fibonacci_sphere(samples=samples, fov=fov)

    phis = phis[thetas <= np.pi / 2]
    thetas = thetas[thetas <= np.pi / 2]
    thetas = (thetas - np.pi) % (2 * np.pi) - np.pi
    phis = (phis + np.pi) % (2 * np.pi) - np.pi

    luminance, dop, aop = sky.lum_aop_dop_from_position(theta_sensor=thetas, phi_sensor=phis, noise=0., eta=None, uniform_polariser=False)

    vmax = np.ceil(np.max(luminance))

    # theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    ax.scatter(phis, thetas, s=20, c=luminance, marker='.', cmap='Blues_r', vmin=0, vmax=vmax)
    # ax.scatter(phis, thetas, s=20, c=luminance, marker='.', cmap='Blues_r')
    if sun is not None:
        sun_azi, sun_ele = sun2azi_ele(sun)
        ax.scatter(sun_azi, sun_ele, s=100, edgecolor='black', facecolor='yellow')
    # zenith location?
    if sensor is not None:
        # todo - change to theta_t when tilt is implemented
        print('*************')
        print(sensor.phi)
        print(sensor.theta)
        print('*************')
        # sensor_luminance, sensor_dop, sensor_aop = sky.lum_aop_dop_from_position(theta_sensor=sensor.theta, phi_sensor=sensor.phi, noise=0., eta=None,
        #                                                        uniform_polariser=False)
        sensor_luminance, sensor_dop, sensor_aop = sky(theta_sensor=sensor.theta, phi_sensor=sensor.phi, noise=0., eta=None,
                                                               uniform_polariser=False)
        print('lum: {} dop: {} aop: {}'.format(sensor_luminance, sensor_dop, sensor_aop))
        ax.scatter(sensor.phi, sensor.theta, s=100, c=sensor_luminance, edgecolor='black', cmap='Blues_r', vmin=0, vmax=vmax)

    ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')

    ax.set_yticks([0, np.pi/4, np.pi/2])
    ax.set_yticklabels(['$0$', '$\pi/4$', '$\pi$/2'], fontweight='bold')
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    # pc = ax.pcolormesh(phis[:, np.newaxis], thetas[:, np.newaxis], luminance[:, np.newaxis])
    # fig.colorbar(pc)

    if show:
        plt.show()
        # ax = plt.subplot(111)
        # ax.scatter(phis, thetas)
        # plt.show()
    return ax

# edinburgh_sky, edinburgh_sun = get_edinburgh_sky()
# visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=compass)




from sphere import azidist
class Trial(object):
    def __init__(self, sensor_class=None, sky_class=None, network_class=None,
                # experiment parameters
                samples=3
                 ):
             # nb_pol=60, omega=56, sigma=np.deg2rad(13), shift=np.deg2rad(40),
             # nb_sol=8, nb_tcl=8, noise=0.,
             #
             # fibonacci=False,
             # uniform_polariser=False,
             #
             # # single evaluation
             # sun_azi=None, sun_ele=None,
             #
             # # data parameters
             # tilting=True, ephemeris=False,
             # samples=1000, snap=False, verbose=False

        if sensor_class is None:
            op_units = 8
            ele = np.deg2rad(45)
            rho = 5.4   # acceptance angle
            omega = 56  # fov of sensor - unused now?
            self.sensor = SimpleCompass(op_units=op_units, ele=ele, omega=omega, rho=rho)
            print('sensor thetas init: {}'.format(self.sensor.theta))
        else:
            self.sensor = sensor_class

        if sky_class is None:
            print('Using Edinburgh sky position')
            # todo - why doesn't initialising a blank sky work?
            self.sky = Sky()
            # self.sky = Sky()
            # date_str = '2021/06/06 12:00'
            # sun location
            # sun = ephem.Sun()
            # sun.date = date_str
            # sun.compute(sky)  # compute sun position relative to observer
            # # self.sky()
            # self.sky, edinburgh_sun = get_edinburgh_sky(date_str='2021/06/01 12:00')
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

    def run(self, sun_thetas=None, sun_phis=None):

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
            luminance, dop, aop = self.sky(theta=self.sensor.theta,
                                             phi=self.sensor.phi, noise=0., eta=None,
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




# sun_phis = np.deg2rad(45) # 3.057426929473877

date_str = '2021/06/01 12:00'
sky, sun = get_edinburgh_sky(date_str=date_str)
sun_azi, sun_ele = sun2azi_ele(sun)

#################### single trial, 8 pol sensors
# t = Trial()
# # err_deg = t.run(sun_thetas=[np.pi/4,], sun_phis=[np.pi/1.5,])
# test1_err_deg = t.run(sun_thetas=[sun_ele,], sun_phis=[sun_azi,])
# print(test1_err_deg)
# # visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=t.sensor)
# # fig, ax = t.network.plot_weight_states()
# # plt.show()
# # del t

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




# # omega = 56
# # theta, phi = fibonacci_sphere(nb_pol=60, float(omega))
# sensor = DRA(op_units=200, n=200)
# t = Trial(sensor_class=sensor)
# test2_err_deg = t.run(sun_thetas=[0.5813175002720694,], sun_phis=[sun_phis,])
# print(test1_err_deg)
# # visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=t.sensor)
