from matplotlib import pyplot as plt
import numpy as np

from compoundeye.geometry import angles_distribution, fibonacci_sphere
from small_sensor.trials import Trial
from small_sensor.sensor import FibonacciCompass, SimpleCompass, SphericalDistributionCompass


def generate_random_sun_positions(samples, fov=90, seed=None,
                                  theta_lower_deg=5, theta_upper_deg=90,
                                  phi_lower_deg=0, phi_upper_deg=360):

    # generate random sun positions
    theta_lower_rad = np.deg2rad(theta_lower_deg)
    theta_upper_rad = np.deg2rad(theta_upper_deg)
    phi_lower_rad = np.deg2rad(phi_lower_deg)
    phi_upper_rad = np.deg2rad(phi_upper_deg)

    if seed:
        np.random.seed(seed)
    sun_thetas = np.random.uniform(theta_lower_rad, theta_upper_rad, samples)
    sun_phis = np.random.uniform(phi_lower_rad, phi_upper_rad, samples)

    # sun_thetas = (sun_thetas - np.pi) % (2 * np.pi) - np.pi
    # sun_phis = (sun_phis + np.pi) % (2 * np.pi) - np.pi
    return sun_thetas, sun_phis

def generate_evenly_spaced_sun_distribution(samples=500, fov=90):

    sun_thetas, sun_phis = fibonacci_sphere(samples=samples, fov=fov)

    return sun_thetas, sun_phis

class EvaluateSensor(object):

    def __init__(self, sample_qty=1000, trial=None, sample_method='fibonacci', seed=None):

        # parse arguments
        if not trial:
            trial = Trial()

        self.trial = trial

        sample_methods = ['random', 'fibonacci']
        if sample_method not in sample_methods:
            raise ValueError("Invalid sample method {}. Expected one of: {}".format(sample_method, sample_methods))
        self.sample_method = sample_method

        self.sample_qty = sample_qty
        self.azimuth_error = np.zeros(self.sample_qty)
        self.confidences = np.zeros(self.sample_qty)

        if self.sample_method == 'random':
            self.sun_thetas, self.sun_phis = generate_random_sun_positions(samples=self.sample_qty, seed=seed)
        elif self.sample_method == 'fibonacci':
            self.sun_thetas, self.sun_phis = generate_evenly_spaced_sun_distribution(samples=self.sample_qty, fov=90)

    def evaluate_single(self, print_output=False, theta_tilt=0., phi_tilt=0.):

        # self.azimuth_error = np.zeros(self.sample_qty)
        # self.confidences = np.zeros(self.sample_qty)

        for idx, (sun_theta, sun_phi) in enumerate(zip(self.sun_thetas, self.sun_phis)):

            # self.azimuth_error[idx] = self.trial.run(sun_theta=sun_theta, sun_phi=sun_phi, theta_tilt=theta_tilt,
            #                                          phi_tilt=phi_tilt)
            self.azimuth_error[idx] = self.trial.run(sun_theta=sun_theta, sun_phi=sun_phi)
            self.confidences[idx] = self.trial.tau_pred

        self.mean_azimuth_error = np.mean(self.azimuth_error)
        self.std_azimuth_error = np.std(self.azimuth_error)
        self.mean_confidence = np.mean(self.confidences)
        self.std_confidence = np.std(self.confidences)

        if print_output:
            print('  Parameter      |   Mean score   |      Std')
            print('  ---------      |  ---------     |  ---------')
            print('Azimuth error          {:.3f}            {:.3f}'.format(self.mean_azimuth_error, self.std_azimuth_error))
            print(' Confidence            {:.3f}            {:.3f}'.format(self.mean_confidence, self.std_confidence))
            print('\n')

    def plot_evlaution(self):
        plt.scatter(np.rad2deg(self.sun_thetas), self.azimuth_error, label='Actual error')
        plt.xlabel('Sun elevation')
        plt.ylabel('Azimuth error')
        plt.scatter(np.rad2deg(self.sun_thetas), self.confidences, label='Confidence value')
        plt.grid()
        plt.legend()
        plt.show()

def trial_different_sensor_numbers(samples=500):


    mean_azimuth_errors = []
    azimuth_errors = []
    pol_units = []

    seed = 2021
    for idx in range(2, 21):

        simple_compass = SimpleCompass(op_units=idx)
        trial = Trial(sensor_class=simple_compass)
        evaluate_sensor = EvaluateSensor(trial=trial, sample_qty=samples, sample_method='random', seed=seed)
        evaluate_sensor.evaluate_single()

        azimuth_errors.append(evaluate_sensor.azimuth_error)
        mean_azimuth_errors.append(evaluate_sensor.mean_azimuth_error)
        pol_units.append(idx)


    print(mean_azimuth_errors)

    fig, ax = plt.subplots()
    pos = np.array(range(len(azimuth_errors))) + 1
    bp = ax.boxplot(azimuth_errors, sym='k+', positions=pos)
    # plt.yscale('log')
    ax.set_xticklabels(pol_units)
    ax.set_xlabel('Number of Pol units')
    ax.set_ylabel('Azimuth error')
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    plt.title('Azimuth with errors for {} random sun locations'.format(samples))
    # plt.scatter(mean_azimuth_errors, label='Mean azimuth error')
    # plt.xlabel('Mean error')
    # plt.ylabel('Azimuth error')
    # plt.scatter(np.rad2deg(evaluate_sensor.sun_thetas), evaluate_sensor.confidences, label='Actual error')
    plt.grid()
    plt.show()

if __name__ == '__main__':


    spherical_compass = SphericalDistributionCompass()
    trial = Trial(sensor_class=spherical_compass)
    evaluate_sensor = EvaluateSensor(trial=trial, sample_qty=500, sample_method='random')
    evaluate_sensor.evaluate_single(print_output=True)
    evaluate_sensor.plot_evlaution()

    simple_compass = SimpleCompass()
    trial = Trial(sensor_class=simple_compass)
    evaluate_sensor = EvaluateSensor(trial=trial, sample_qty=500, sample_method='random')
    evaluate_sensor.evaluate_single(print_output=True)
    evaluate_sensor.plot_evlaution()

    trial_different_sensor_numbers()
