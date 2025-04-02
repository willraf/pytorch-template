""" Module generate_data

Contains code to generate synthetic sEMG data as the convolutional sum of simulated spike trains 

Randomly generated spike trains are convolved with a family of filters.
These filtered signals are the summed to get simulated sEMG signals. 

Usage:

Authors:

Date: 26/11/2024

TODO:
    - Name of period variable (currently sampling_period)
        - dt
        - time_period
        - sampling_period
        - period
    - Change sampling_period to sampling_frequency?
    - NOTE on matrix convention: time is LAST dimension, unlike ML standard (first dim)
    - Watch for overflow errors on spike indexes.
    - Change parameters to be more centered around time than time steps. 
        Time might mean more to a user e.g. length of filter in time.
        Then do conversion hidden
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(
    sources=1, 
    channels=1,
    duration=1., 
    sampling_frequency=1000,
    filter_type="g"
    ):
    """
    
    Args:
        sources (int): The number of sources/MUs to simulate (N)
        channels (int): The number of sEMG channels to simulate (M)
        duration (float): The number of samples in time to simulate (Q)
        sampling_period (int): The sampling period in milliseconds
        filter_type (str): The family of filters to apply to spike trains

    Returns:
        X, y : Synthetic data
            - X (np.ndarray): Tensor of shape (m, q) q samples in time of sEMG signal with m channels 
            - y (np.ndarray): Tonsor shape (n, q) activation of n MUs over q time steps
    """
    sampling_period = 1/sampling_frequency


    # Generate spike trains
    firing_rates = np.array([10, 20, 50])
    firing_rates = np.random.randint(5, 100, (sources))
    

    spike_times = multivariate_poisson_spike_trains(firing_rates, duration=duration)

    spike_matrix = multivariate_spike_train_to_discrete(spike_times=spike_times, duration=duration, sampling_period=sampling_period)

    # Change shape to have time on second axis. Shape (N, Q)
    spike_matrix = spike_matrix.T

    T = spike_matrix.shape[1]


    # Generate filters
    L = 30

    # Define parameters for each filter 
    # i^th channel j^th source
    mu_H = np.array([[0.005, 0.01],
                    [0.015, 0.0075],
                    [0.01, 0.008]])

    sigma_H = np.array([[0.01, 0.015],
                        [0.01, 0.01], 
                        [0.01, 0.01]])

    # Shape (m, n, L)
    H = generate_filter_tensor(mu_H, sigma_H, L, sampling_period)

    # Reshape tensors

    t_indices = np.arange(T) 
    l_indices = np.arange(L)

    # Compute the shifted indices (broadcasting L over T)
    # Matrix of indeces with X_ij = j - i
    shifted_indices = t_indices[None, :] - l_indices[:, None]

    valid_mask = shifted_indices >= 0

    N = sources
    M = channels

    s = spike_matrix

    S = np.zeros((N, L, T))

    for j in range(N):  
        S[j] = np.where(valid_mask, s[j, shifted_indices], 0)


    H_prime = H.reshape(M, N*L)
    S_prime = S.reshape(N*L, T)

    x = H_prime @ S_prime
    print(x)
    print(x.shape)
    print(s.shape)

    # Apply convolution


def generate_filter_tensor(mu_H, sigma_H, L, dt):
    """
    Generate a tensor of filters.

    Filter for the i_th channel j_th source is defined by parameters mu_H_ij and sigma_H_ij

    Args:
        mu_H (np.ndarray): Shape (m, n). Mean parameter of filters
        sigma_H (np.ndarray): Shape (m, n). std of filters
        L (int): (max) length of the finite impulse response filters in time steps
    """
    m, n = mu_H.shape[0], mu_H.shape[1]


    if mu_H.shape != sigma_H.shape:
        print(f'mu_H shape: {mu_H.shape} \t sigma_H shape: {sigma_H.shape}')
        raise Exception("Incompatable tensor shape mu_H, sigma_H")

    # repeat parameters to match shape of filter matrix H
    mu_H = mu_H.reshape(m, n, 1)
    mu_H = np.tile(mu_H, (1, 1, L))

    sigma_H = sigma_H.reshape(m, n, 1)
    sigma_H = np.tile(sigma_H, (1, 1, L))

    # Create matrix of "time step" indexes
    x = np.arange(L).reshape(1, 1, L)
    x = np.tile(x, (m, n, 1))

    # Convert to the real time of each time step
    times = x * dt

    # Sample filter from the gaussian using parameters
    return gaussian(mu_H, sigma_H, times)


def gaussian_spike_train(mu_count, duration, fs, Tmean=0.1, Tstd=0.03):
    ''' BY DIMITRIOS

    Generate motor unit spike trains.'''
    spts = np.zeros((mu_count, int(fs * duration)))
    dts = []
    
    # For every independent MU
    for mu_idx in range(mu_count):
        times = np.random.normal(loc=Tmean, scale=Tstd, size=int(duration / Tmean))
        times = np.cumsum(times)  # cumulative sum of firing times
        dts.append(times)  # become discharge times in seconds
        times = (fs * times[times <= duration]).astype(int)
        spts[mu_idx, times] = 1  # set all firing time values to 1
    
    return spts, dts


def poisson_spike_train(firing_rate, duration):
    """
    Simulate a spike train using a poisson process

    Args:
        firing_rate (float): expected spikes per second (Hz)/rate parameter (lambda)
        duration (float): Time duration of simulation (s)

    Returns:
        np.ndarray: Array of spike train time stamps
    """
    scale_parameter = 1 / firing_rate # The expected time period between spikes (ms)

    spike_times = []
    t = 0 

    while t < duration:
        spike_gap = np.random.exponential(scale_parameter)

        t += spike_gap

        if t < duration:
            spike_times.append(t)
            
    return np.array(spike_times)


def spike_train_to_discrete(spike_times, duration, sampling_period):
    """
    Convert spike train time stamps to a discrete binary vector

    Args: 
        spike_times (np.ndarray): Time stamps of spikes (MUAPs)
        duration (float): Duration of simulation (s)
        sampling_period (float): Size of each time bin (s). Equal to time period of sEMG
    
    Returns:
        np.ndarray: Shape (duration/sampling_period,). 
            Discrete binary vector of spike train. 1 at time steps with activation, 0 otherwise.  
    """

    # Convert to indexes of vector 
    spike_indexes = (spike_times // sampling_period).astype(int)

    length = int(np.ceil(duration / sampling_period))
    spike_vector = np.zeros(length, dtype=int)

    spike_vector[spike_indexes] = 1

    return spike_vector


def visualise_spike_train(spike_times):
    """
    Plot an individual spike train

    Args:
        spike_times (np.ndarray): Array of spike time stamps
    """

    y = np.repeat(1, spike_times.shape[0])

    plt.plot(spike_times, y, "o")
    plt.vlines(x=spike_times, ymin=0, ymax=1, color='blue', linestyle='-', label='Vertical line')
    plt.ylim(0, 3)

    plt.xlabel("Time (s)")

    plt.show()


def multivariate_poisson_spike_trains(firing_rates, duration):
    """
    Simulate a number of spike trains, each with different firing rates. 

    Args:   
        firing_rates (np.ndarray): Firing rates of each source/MU. Length n
        duration (float): Time duration of simulation (s)

    Returns:
        np.ndarray: Shape (s, N). Time stamps of spikes (MUAPs)
            - s: Max number of spikes
            - N: Number of sources
    """
    scale_parameters = 1 / firing_rates # The expected time period between spikes (ms)

    spike_times = []
    times = np.zeros_like(firing_rates, dtype="float64")

    while np.any(times < duration):
        spike_gaps = np.random.exponential(scale_parameters)

        times += spike_gaps

        times[times >= duration] = None

        if not np.all(np.isnan(times)):
            spike_times.append(times.copy())


    return np.array(spike_times)


def multivariate_spike_train_to_discrete(spike_times, duration, sampling_period):
    """
    Convert matrix of spike train time stamps to a discrete binary matrix

    Args: 
        spike_times (np.ndarray): Shape (s, N). Time stamps of spikes (MUAPs)
            - s: Max number of spikes
            - N: Number of sources
        duration (float): Duration of simulation (s)
        sampling_period (float): Size of each time bin (s). Equal to time period of sEMG
    
    Returns:
        np.ndarray: Shape (Discrete binary matrix of spike train. 
            1 at time steps with activation, 0 otherwise.  
    """

    spike_times = np.where(np.isnan(spike_times), -1, spike_times)

    # Convert to indexes of vector 
    spike_indexes = (spike_times // sampling_period).astype(int)

    length = int(np.ceil(duration / sampling_period))
    num_sources = spike_times.shape[1]

    spike_matrix = np.zeros((length, num_sources), dtype=int)

    # Filter out invalid indexes (e.g., -1 from NaN values)
    for source in range(num_sources):
        valid_indexes = spike_indexes[:, source]
        valid_indexes = valid_indexes[valid_indexes >= 0]  # Keep only valid indexes
        spike_matrix[valid_indexes, source] = 1

    return spike_matrix


def generate_gaussian_filter(mu, sigma, L, dt):
    """
    Generate a discrete gaussian filter

    A finite impulse filter which represent temporal dispersion caused by volume conduction
    and distances between source (MU fibres) and sEMG electrode.

    Args:
        mu (float): Mean of gaussian. 
            The time lag for which the greatest proportion of the source is measured on the skin surface.
            Should likely be greater than 0
        sigma (float): std of the gaussian. 
            Controls how much the propogation to the skins surface spreads the action potential spike
        L (int): Length of the finite impulse filter (number of time steps).
        dt (float): The difference in time between each time step

    Returns:
        np.ndarray: Shape (L,). The discrete gaussian filter. 
    """

    x = np.arange(L)

    # The real time of each time step
    times = x * dt
    return gaussian(mu, sigma, times)


def gaussian(mu, sigma, x):
    A = 1 / (np.sqrt(2 * np.pi * sigma * sigma))
    B = ((x - mu)**2) / (2 * sigma**2)
    return A * np.exp(-B)


def single_source_channel():
    """
    Initial prototype function to generate and visualise data for a single source and channel
    """
    # Number of time steps to simulate
    T = 1000

    # Length of filter
    L = 30

    # Sampling period
    dt = 0.001

    # Filter parameters (in seconds)
    mu = 0.01
    sigma = 0.01

    firing_rate = 5

    time_duration = T*dt

    # Generate spike train
    spike_times = poisson_spike_train(firing_rate, time_duration)
    spike_vector = spike_train_to_discrete(spike_times, time_duration, dt)

    # Apply filter
    y = np.zeros(T)

    for t in range(T):
        for l in range(min(t, L)):
            y[t] += (gaussian(mu, sigma, l*dt) * spike_vector[t - l])

    
    # Display output
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    # Spike train
    spike_y = np.repeat(1, spike_times.shape[0])
    ax1.plot(spike_times, spike_y, "o", color="blue")
    ax1.vlines(x=spike_times, ymin=0, ymax=1, color='blue', linestyle='-', label='Vertical line')
    ax1.set_ylim(0, 3)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Spike Train")

    # Filtered
    x = np.arange(0, time_duration, dt)
    ax2.plot(x, y, 'o', color="orange")
    ax2.set_title("Filtered Signal")

    ax2.sharex(ax1)

    # Filter
    x = np.arange(L) * dt
    y = gaussian(mu, sigma, x)
    ax3.plot(x, y, 'o', color="green")
    ax3.set_title("Guassian Filter")


    fig.tight_layout()

    # Show the plot
    plt.show()


def test_poisson_spike_train():

    spike_times = poisson_spike_train(50, 1)
    visualise_spike_train(spike_times)
    spike_vector = spike_train_to_discrete(spike_times, 1, 0.001)
    print(spike_vector)

def test_multivariate_possion_spike_trains():

    firing_rates = np.array([10, 20, 50])

    spike_times = multivariate_poisson_spike_trains(firing_rates, 1)

    spike_matrix = multivariate_spike_train_to_discrete(spike_times, 1, 0.001)
    print(spike_matrix)

def test_generate_data():
    generate_data()

def test_gaussian():
    x = np.arange(0, 10, 0.1)
    y = gaussian(5, 2, x)

    plt.plot(x, y, 'o')
    plt.show()

def test_generate_gaussian_filter():
    mu = 5
    sigma = 1
    L = 150
    dt = 0.1

    total_time = L*dt

    y = generate_gaussian_filter(mu, sigma, L, dt)
    x = np.arange(0, total_time, dt)
    plt.plot(x, y, 'o')
    plt.show()


if __name__ == "__main__":


    generate_data(sources=2, channels=3)
















