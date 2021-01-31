import numpy as np
from matplotlib import pyplot as plt

from small_sensor.sensor import SphericalDistributionCompass
from environment.sky import sun2azi_ele, Sky
from sphere.transform import tilt


def visualise_luminance(sky, show=True, ax=None):

    if not ax:
        plt.figure("Luminance", figsize=(4.5, 4.5))
        ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    ax.scatter(sky.phi, sky.theta, s=20, c=sky.Y, marker='.', cmap='Blues_r', vmin=0, vmax=6)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    if show:
        plt.show()
    return ax


def visualise_degree_of_polarisation(sky, show=True, ax=None):

    if not ax:
        plt.figure("degree-of-polarisation", figsize=(4.5, 4.5))
        ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    print(theta_s, phi_s)
    ax.scatter(sky.phi, sky.theta, s=10, c=sky.DOP, marker='.', cmap='Greys', vmin=0, vmax=1)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    if show:
        plt.show()
    return ax


def visualise_angle_of_polarisation(sky, show=True, ax=None):

    if not ax:
        plt.figure("angle-of-polarisation", figsize=(4.5, 4.5))
        ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    print(theta_s, phi_s)
    ax.scatter(sky.phi, sky.theta, s=10, c=sky.AOP, marker='.', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    # ax.scatter(sky.phi_t + np.pi, sky.theta_t, s=200, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    if show:
        plt.show()
    return ax


def visualise_lum_angle_degree(sky):
    import matplotlib.pyplot as plt
    plt.figure("Sky model", figsize=(9, 4.5))

    ax1 = plt.subplot(131, polar=True)
    visualise_luminance(sky, ax=ax1, show=False)
    ax2 = plt.subplot(132, polar=True)
    visualise_degree_of_polarisation(sky, ax=ax2, show=False)
    ax3 = plt.subplot(133, polar=True)
    visualise_degree_of_polarisation(sky, ax=ax3, show=False)

    plt.show()


def visualise_luminance_2(sky, sun=None, sensor=None, show=True, ax=None):

    fig = plt.figure()
    if not ax:
        plt.figure("Luminance", figsize=(4.5, 4.5))
        # ax = plt.subplot(111, polar=True)
        ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # generate samples to plot sky
    samples = 1000
    fov = 180
    fib_compass = SphericalDistributionCompass(op_units=1000, fov=160, rho=5.4)
    # thetas, phis = fibonacci_sphere(samples=samples, fov=fov)
    # phis = phis[thetas <= np.pi / 2]
    # thetas = thetas[thetas <= np.pi / 2]
    # thetas = (thetas - np.pi) % (2 * np.pi) - np.pi
    # phis = (phis + np.pi) % (2 * np.pi) - np.pi

    luminance, dop, aop = sky(sensor=fib_compass, noise=0., eta=None, uniform_polariser=False)

    vmax = np.ceil(np.max(luminance))

    # theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    ax.scatter(fib_compass.phi, fib_compass.theta, s=20, c=luminance, marker='.', cmap='Blues_r', vmin=0, vmax=vmax)
    # ax.scatter(phis, thetas, s=20, c=luminance, marker='.', cmap='Blues_r')
    if sun is not None:
        sun_azi, sun_ele = sun2azi_ele(sun)
        ax.scatter(sun_azi, sun_ele, s=100, edgecolor='black', facecolor='yellow')
    # zenith location?
    if sensor is not None:
        # todo - change to theta_t when tilt is implemented
        # print('*************')
        # print(sensor.phi)
        # print(sensor.theta)
        # print('*************')
        # sensor_luminance, sensor_dop, sensor_aop = sky.lum_aop_dop_from_position(theta_sensor=sensor.theta, phi_sensor=sensor.phi, noise=0., eta=None,
        #                                                        uniform_polariser=False)
        sensor_luminance, sensor_dop, sensor_aop = sky(sensor=sensor, noise=0., eta=None, uniform_polariser=False)

        # print('lum: {} dop: {} aop: {}'.format(sensor_luminance, sensor_dop, sensor_aop))
        ax.scatter(sensor.phi, sensor.theta, s=100, c=sensor_luminance, edgecolor='black', cmap='Blues_r', vmin=0, vmax=vmax)
        ax.scatter(sensor.phi_t + np.pi, sensor.theta_t, s=200, edgecolor='black', facecolor='greenyellow')



    ax.set_yticks([0, np.pi/4, np.pi/2])
    ax.set_yticklabels([r'$0\circ$', r'$45\circ$', r'$90\circ$'], fontweight='bold')
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


if __name__ == "__main__":

    from environment.sky import get_edinburgh_sky

    sky = get_edinburgh_sky()
    visualise_luminance_2(sky=sky, sensor=SphericalDistributionCompass(op_units=20))


