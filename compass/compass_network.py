import matplotlib.pyplot as plt
import numpy as np

from environment import eps, spectrum_influence, spectrum


class Network(object):

    def __init__(self, nb_pol, nb_sol=8, nb_tcl=8,
                 ephemeris=False,
                 ):

        self.nb_sol = nb_sol
        self.nb_tcl = nb_tcl
        self.nb_pol = nb_pol

        # network layers
        self.r_pol = np.tile(np.nan, nb_pol)
        self.r_sol = np.tile(np.nan, nb_sol)
        self.w_tcl = np.tile(np.nan, nb_tcl)
        self.r_tcl = np.tile(np.nan, nb_tcl)

        # initialise other fields
        self.FFT_r_tcl = np.nan
        self.a_pred = np.nan
        self.tau_pred = np.nan

        # computational model parameters
        self.phi_sol = np.linspace(0., 2 * np.pi, self.nb_sol, endpoint=False) #- (np.pi / 4)  # SOL preference angles
        self.phi_tcl = np.linspace(0., 2 * np.pi, self.nb_tcl, endpoint=False) #- (np.pi / 4)  # TCL preference angles

    def compute(self, sensor, luminance, dop, aop, do_tilt=False):

        # Input (POL) layer -- Photo-receptors
        print('luminance ALL: {}'.format(luminance))
        luminance = spectrum_influence(v=luminance, wl=spectrum["uv"])
        print('luminance UV: {}'.format(luminance))

        # log for plotting
        self.latest_luminance = luminance
        self.latest_dop = dop
        self.latest_aop = aop

        alpha = (sensor.phi + np.pi / 2) % (2 * np.pi) - np.pi
        alpha_ = alpha # todo - this will be the tilt angle
        # alpha_ = 0 # todo - this will be the tilt angle
        # todo - how does this relate to the paper formula (eq1 p7)?
        # s_par = s_parallel
        # s_per = s_perpendicular
        # alpha = orientation of the primary axis of one of the sensor units?
        # alpha_ = tilted alpha
        s_par = luminance * (np.square(np.sin(aop - alpha_)) + np.square(np.cos(aop - alpha_)) * np.square(1. - dop))
        s_per = luminance * (np.square(np.cos(aop - alpha_)) + np.square(np.sin(aop - alpha_)) * np.square(1. - dop))
        r_par, r_per = np.sqrt(s_par), np.sqrt(s_per)
        r_op, r_po = r_par - r_per, r_par + r_per
        self.r_pol = r_op / (r_po + eps)
        print('rpol: {}'.format(self.r_pol))

        # todo - add in tilt layer
        # # Tilting (SOL) layer
        if do_tilt:
            # d_gate = (np.sin(shift - theta) * np.cos(theta_t) +
            #           np.cos(shift - theta) * np.sin(theta_t) *
            #           np.cos(phi - phi_t))
            # gate = .5 * np.power(np.exp(-np.square(d_gate) / (2. * np.square(sigma))), 1)
            pass

        # alpha = (sensor.phi + np.pi / 2) % (2 * np.pi) - np.pi
        # alpha = (sensor.phi + np.pi / 2) % (2 * np.pi) - np.pi

        z_pol = float(self.nb_sol) / float(self.nb_pol)
        w_sol = z_pol * np.sin(alpha[:, np.newaxis] - self.phi_sol[np.newaxis]) # * gate[:, np.newaxis] # <- todo required for gating
        self.r_sol = self.r_pol.dot(w_sol)
        # self.r_sol = self.r_pol
        # print('rsol: {}'.format(self.r_sol))

        # Output (TB1) layer
        # eph = (a - np.pi / 3) if self.ephemeris else 0.  # nb: a = phi_s
        eph = 0
        self.w_tcl = float(self.nb_tcl) / float(2. * self.nb_sol) * np.cos(self.phi_tcl[np.newaxis] - self.phi_sol[:, np.newaxis] + eph)

        # r_tb1 = r_cl1.dot(w_tb1)
        self.r_tcl = self.r_sol.dot(self.w_tcl)

        # print('w_tcl: {}'.format(self.w_tcl))
        # print('r_tcl: {}'.format(self.r_tcl))

        return self.r_tcl

    def decode_response(self):
        # decode response - FFT
        self.FFT_r_tcl = self.r_tcl.dot(np.exp(-np.arange(self.nb_tcl) * (0. + 1.j) * 2. * np.pi / float(self.nb_tcl)))    # Fast fourier transform of response
        self.azimuth_pred = (np.pi - np.arctan2(self.FFT_r_tcl.imag, self.FFT_r_tcl.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)
        self.tau_pred = np.absolute(self.FFT_r_tcl)   # confidence of the prediction

    def plot_weight_states(self, label_font_size=11, unit_font_size=10, colormap='viridis'):

        # sources = ['TL2', 'CL1', 'TB1', 'TB1', 'TN', 'TB1', 'TB1', 'CPU4', 'CPU4',
        #            'CPU4', 'Pontin', 'Pontin']
        # targets = ['CL1', 'TB1', 'TB1', 'CPU4', 'CPU4', 'CPU1a', 'CPU1b', 'CPU1a',
        #            'CPU1b', 'Pontin', 'CPU1a', 'CPU1b']
        # ticklabels = {'TL2': range(1, 17),
        #               'CL1': range(1, 17),
        #               'TB1': range(1, 9),
        #               'TN': ['L', 'R'],
        #               'CPU4': range(1, 17),
        #               'Pontin': range(1, 17),
        #               'CPU1a': range(2, 16),
        #               'CPU1b': range(8, 10)}


        weights = [self.latest_luminance, self.latest_dop, self.latest_aop, self.r_pol, self.r_sol, self.w_tcl, self.r_tcl]
        # weights = [self.r_pol, self.r_sol, self.w_tcl, self.r_tcl]
        qty = len(weights)
        fig, ax = plt.subplots(qty, 1, figsize=(12, 16))

        width = 1
        vmax = 1
        vmin = -1
        for i in range(qty):
            cax = ax[i]   # ax[i / width] # [i % width]
            if weights[i].ndim == 1:
                weights[i] = np.expand_dims(weights[i], axis=0)
            p = cax.pcolor(weights[i], cmap=colormap) #, vmin=vmin, vmax=vmax)
            p.set_edgecolor('face')
            cax.set_aspect('equal')

            cax.set_xticks(np.arange(weights[i].shape[1]) + 0.5)
            # cax.set_xticklabels(ticklabels[sources[i]])

            cax.set_yticks(np.arange(weights[i].shape[0]) + 0.5)
            # cax.set_yticklabels(ticklabels[targets[i]])

            # if i == 1:
            #     cax.set_title(sources[i] + ' to ' + targets[i], y=1.41)
            # else:
            #     cax.set_title(sources[i] + ' to ' + targets[i])
            #
            # cax.set_xlabel(sources[i] + ' cell indices')
            # cax.set_ylabel(targets[i] + ' cell indices')
            cax.tick_params(axis=u'both', which=u'both', length=0)

            fig.colorbar(p, ax=cax)

        # cbax = fig.add_axes([0.8, 0.2, 0.02, 0.3])
        # m = cm.ScalarMappable(cmap=colormap)
        # m.set_array(np.linspace(vmin, vmax, 100))
        # cb = fig.colorbar(m, cbax, ticks=[-1, -0.5, 0, 0.5, 1])
        # cb.set_label('Connection Strength', labelpad=-50)
        # cb.ax.set_yticklabels(['-1.0 (Inhibition)', '-0.5', '0.0', '0.5',
        #                        '1.0 (Excitation)'])
        plt.tight_layout()
        return fig, ax