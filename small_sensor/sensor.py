from matplotlib import pyplot as plt
import numpy as np

from environment import Environment, spectrum, spectrum_influence, eps
from sphere.transform import tilt
from sphere import angdist

from small_sensor.utils import wrap_to_pi

class GenericPolSensor(object):

    def __init__(self, thetas=None, phis=None, rho=5.4, nb_pr=8, theta_c=0., phi_c=0.):
        """
        :param n: number of ommatidia
        :type n: int
        :param nb_pr: number of photo-receptors per ommatidium
        :type nb_pr: int
        :param rho: acceptance angle (degrees)
        :type rho: float, np.ndarray
        """
        # todo: make op_units and n a single entity
        self.op_units = np.size(thetas)

        self.theta = wrap_to_pi(thetas - np.pi)  # todo - why - pi? (sensor doesn't work without this)
        self.phi = wrap_to_pi(phis + np.pi)      # todo - why + pi? (sensor doesn't work without this)
        # set the orientation of the pol sensor to be perpendicular to that sensor's azimuth
        self.alpha = wrap_to_pi(self.phi + np.pi/2)   # todo - rename alpha

        # create transformation matrix of the perceived light with respect to the optical properties of the eye
        rho = np.deg2rad(rho)
        self.rho = rho if rho.size == self.op_units else np.full(self.op_units, rho)
        sph = np.array([self.theta, self.phi])
        i1, i2 = np.meshgrid(np.arange(self.op_units), np.arange(self.op_units))
        i1, i2 = i1.flatten(), i2.flatten()
        sph1, sph2 = sph[:, i1], sph[:, i2]
        d = np.square(angdist(sph1, sph2).reshape((self.op_units, self.op_units)))
        sigma = np.square([self.rho] * self.op_units) + np.square([self.rho] * self.op_units).T
        self._rho_gaussian = np.exp(-d/sigma)
        self._rho_gaussian /= np.sum(self._rho_gaussian, axis=1)

        # todo - what are theta_c and phi_c
        self._theta_c = theta_c
        self._phi_c = phi_c
        # set tilt angle
        self._theta_t = 0.
        self._phi_t = 0.
        self.__r = np.full((self.op_units), np.nan)

        # setup the sensor spectral sensitivity - just UV by default
        self.rhabdom = np.array([[spectrum["uv"]] * self.op_units] * nb_pr)

        # setup microvilli
        self.mic_l = np.zeros((nb_pr, self.op_units), dtype=float)  # local (in the ommatidium) angle of microvilli
        self.mic_a = (self.phi + np.pi / 2) % (2 * np.pi) - np.pi   # global (in the compound eye) angle of mictovilli
        self.mic_p = np.zeros((nb_pr, self.op_units), dtype=float)  # polarisation sensitivity of microvilli
        self.mic_l = np.array([[0., np.pi/2, np.pi/2, np.pi/2, 0., np.pi/2, np.pi/2, np.pi/2]] * self.op_units).T
        self.mic_p[:] = 1.
        self.__r_op = np.full(self.op_units, np.nan)
        self.__r_po = np.full(self.op_units, np.nan)
        self.__r_pol = np.full(self.op_units, np.nan)

        self.__is_called = False

    def __call__(self, env, *args, **kwargs):
        """
        :param env: the environment where the photorectors can percieve light
        :type env: Environment
        :param args: unlabeled arguments
        :type args: list
        :param kwargs: labeled arguments
        :type kwargs: dict
        :return:
        """
        env.theta_t = self.theta_t
        env.phi_t = self.phi_t

        _, alpha = tilt(self.theta_t, self.phi_t + np.pi, theta=np.pi / 2, phi=self.mic_a)
        y, p, a = env(self.theta, self.phi, *args, **kwargs)

        # todo:
        # influence of the acceptance angle on the luminance and DOP
        # y = y.dot(self._rho_gaussian)
        # p = p.dot(self._rho_gaussian)

        # influence of the wavelength on the perceived light
        ry = spectrum_influence(y, self.rhabdom)

        s = ry * ((np.square(np.sin(a - alpha + self.mic_l)) +
                   np.square(np.cos(a - alpha + self.mic_l)) * np.square(1. - p)) * self.mic_p + (1. - self.mic_p))
        self.__r = np.sqrt(s)

        self.__r_op = np.sum(np.cos(2 * self.mic_l) * self.__r, axis=0)
        self.__r_po = np.sum(self.__r, axis=0)
        self.__r_pol = self.__r_op / (self.__r_po + eps)
        self.__r_po = 2. * self.__r_po / np.max(self.__r_po) - 1.

        self.__is_called = True

        return self.r_pol
        # return self.__r

    @property
    def theta_t(self):
        return self._theta_t

    @theta_t.setter
    def theta_t(self, value):
        theta_t, phi_t = tilt(self._theta_c, self._phi_c - np.pi, theta=self._theta_t, phi=self._phi_t)
        self._theta_t, phi_t = tilt(self._theta_c, self._phi_c, theta=value, phi=self.phi_t)

    @property
    def phi_t(self):
        return self._phi_t

    @phi_t.setter
    def phi_t(self, value):
        theta_t, phi_t = tilt(self._theta_c, self._phi_c - np.pi, theta=self._theta_t, phi=self._phi_t)
        theta_t, self._phi_t = tilt(self._theta_c, self._phi_c, theta=self.theta_t, phi=value)

    @property
    def r(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r

    @property
    def r_pol(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r_pol

    @property
    def r_po(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r_po

    @property
    def r_op(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r_op

    def __repr__(self):
        # todo
        return "%s(name=%s, n=%d, omega=%f, rho=%f)" % (
            self.__class__.__name__, self.name, self.theta.size, np.max(self.theta) * 2, self.rho[0])

class SimpleCompass(GenericPolSensor):
    def __init__(self, op_units=8, ele=np.deg2rad(45), rho=5.4):

        # a ring of 8 sensors pointing at the sky, 45 degrees up from the horizon
        azis = np.linspace(0, 2 * np.pi, op_units, endpoint=False)
        ele = (np.pi/2) - np.tile(ele, op_units)  # todo - why does elevation start at zenith?

        super().__init__(thetas=ele, phis=azis, rho=rho)


# todo - make sensors for these methods:
#  else:
#             try:
#                 self.theta, self.phi, fit = angles_distribution(n, float(omega))
#             except ValueError:
#                 self.theta = np.empty(0, dtype=np.float32)
#                 self.phi = np.empty(0, dtype=np.float32)
#                 fit = False
#
#             if not fit or n > 100:
#                 self.theta, self.phi = fibonacci_sphere(n, float(omega))

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
        sensor_luminance, sensor_dop, sensor_aop = sky(theta=sensor.theta, phi=sensor.phi, noise=0., eta=None,
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

if __name__ == '__main__':

    op_units = 8
    unit_fov = np.deg2rad(10)
    ele = np.deg2rad(45)
    compass = SimpleCompass(op_units=op_units, ele=ele, omega=unit_fov, rho=unit_fov)
    edinburgh_sky, edinburgh_sun = get_edinburgh_sky()
    visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=compass)