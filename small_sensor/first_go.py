import numpy as np


# from compoundeye.evaluation import evaluate


def sun2lonlat(s, lonlat=False, show=False):
    """
    Returns the position of the sun relative to the obeserver it has been computed against

    s properties:
    s.az — Azimuth east of north
    s.alt — Altitude above horizon
    """
    lon, lat = s.az, s.alt
    colat = (np.pi/2) - lat  # JS - have updated from: np.pi / 2 - lat # todo - understand why lat starts at zenith
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    if show:
        print('Sun:\tLon = %.2f\t Lat = %.2f\t Co-Lat = %.2f' % \
              (np.rad2deg(lon), np.rad2deg(lat), np.rad2deg(colat)))

    if lonlat:  # return the longitude and the latitude in degrees
        return np.rad2deg(lon), np.rad2deg(lat)
    else:  # return the longitude and the co-latitude in radians
        return lon, colat


# def lonlat2theta_phi(lon, lat, theta_z=0, phi_z=0, alt=0):
#     # calculate the angular distance between the sun and every point on the map
#     x, y, z = 0, np.rad2deg(lat), -np.rad2deg(lon)
#     theta_s, phi_s = hp.Rotator(rot=(z, y, x))(theta_z, phi_z)  # todo - remove hp rotator
#     theta_s, phi_s = theta_s % np.pi, (phi_s + np.pi) % (2 * np.pi) - np.pi
#     return theta_s, phi_s







# sun_phis = np.deg2rad(45) # 3.057426929473877
# # omega = 56
# # theta, phi = fibonacci_sphere(nb_pol=60, float(omega))
# sensor = DRA(op_units=200, n=200)
# t = Trial(sensor_class=sensor)
# test2_err_deg = t.run(sun_thetas=[0.5813175002720694,], sun_phis=[sun_phis,])
# print(test1_err_deg)
# # visualise_luminance_2(edinburgh_sky, edinburgh_sun, sensor=t.sensor)
