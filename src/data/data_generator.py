""" Module data_generator

Contains DataGenerator class which generates synthetic sEMG data.

Usage:

Authors:

Date: 26/11/2024

TODO:
    - Potentially transpose output of generate_spike_times_poisson and input to spike_times_to_trains to match later functions
    - Implement visualisation methods
    - Write test cases
    - NOTE on matrix convention: time is LAST dimension, unlike ML standard (first dim)
    - Watch for overflow errors on spike indexes.
    - Add exceptions for non-matching parameter shapes (ValueError)
        - firing_rates, mu_H, sigma_H
    - Within generate_spike_times_poisson, times = np.zeros_like(firing_rates) is error prone
        Too reliant on correct definition of firing_rates (same with mu_H and sigma_H)
    - Make variable names more consistent (e.g. num_sources vs sources)
"""

import numpy as np
import matplotlib.pyplot as plt


class DataGenerator(object):
    """Handels generation of synthetic sEMG data as the convolutional sum of simulated spike trains 

    Randomly generated spike trains are convolved with a family of filters.
    These filtered signals are the summed to get simulated sEMG signals. 
    """

    def __init__(
            self, 
            sources, 
            channels, 
            duration, 
            sampling_frequency,
            filter_type="g"
        ):
        """
        Args:
            sources (int): The number of sources/MUAPs to simulate (n)
            channels (int): The number of sEMG channels to simulate (m)
            duration (float): The length of the simulation in seconds
            sampling_frequency (int): The sampling frequency of sEMG in Hz (also )
            filter_type (str): The family of filters applied
        """
        self.num_sources = sources                                  # n
        self.num_channels = channels                                # m
        self.duration = duration                                    # t?
        self.sampling_frequency = sampling_frequency

        self.sampling_period = 1 / sampling_frequency               # dt

        samples = int(np.ceil(duration / self.sampling_period))     # The number of samples in time to simulate
        self.num_samples = int(samples)                             # T


    def generate_spike_times_poisson(self, firing_rates):
        """
        Simulate a matrix of spike times from multiple sources using a poisson process

        Args:   
            firing_rates (np.ndarray): Shape (n,) Firing rates of each source/MU.

        Returns:
            np.ndarray: Shape (s, n). Time stamps of spikes (MUAPs)
                - s: Max number of spikes
                - n: Number of sources
        """
        scale_parameters = 1 / firing_rates # The expected time period between spikes (s)

        spike_times = []
        times = np.zeros_like(firing_rates, dtype="float64")

        while np.any(times < self.duration):

            # Time gap between spikes randomly distributed according to exponential distribution
            spike_gaps = np.random.exponential(scale_parameters)

            times += spike_gaps

            # Check if any spike times are after simulation duration
            times[times >= self.duration] = None

            if not np.all(np.isnan(times)):
                spike_times.append(times.copy())


        return np.array(spike_times)


    def spike_times_to_trains(self, spike_times):
        """
        Converts a matrix of spike times to a matrix of spike trains (delta functions)

        Args: 
            spike_times (np.ndarray): Shape (s, n). Time stamps of spikes (MUAPs)
                - s: Max number of spikes
                - n: Number of sources

        Returns:
            np.ndarray: Shape (Discrete binary matrix of spike train. 
                1 at time steps with activation, 0 otherwise.  
        """

        # Convert None values of spike times (where simulation for that source has ended)
        # into -1 to prevent type conversion error 
        spike_times = np.where(np.isnan(spike_times), -1, spike_times)

        # Convert to indexes of vector 
        spike_indexes = (spike_times // self.sampling_period).astype(int)

        spike_matrix = np.zeros((self.num_sources, self.num_samples), dtype=int)

        # Filter out invalid indexes (e.g., -1 from NaN values)
        for source in range(self.num_sources):
            valid_indexes = spike_indexes[:, source]
            valid_indexes = valid_indexes[valid_indexes >= 0]  # Keep only valid indexes

            spike_matrix[source, valid_indexes] = 1

        return spike_matrix


    def visualise_spike_trains(self, spike_trains):
        """
        Visualise a number of spike trains

        Args:
            spike_trains (np.ndarray): Shape (n, T)
        """

        fig, axs = plt.subplots(self.num_sources, 1, figsize=(8, 8), sharey=True)

        x = np.arange(self.num_samples) * self.sampling_period

        i = 0
        for source in spike_trains:
            axs[i].plot(x, source, 'o', color="blue")
            spike_times = np.nonzero(source)[0] * self.sampling_period
            axs[i].vlines(spike_times, ymin=0, ymax=1, color='blue', linestyle='-')
            i += 1

        plt.ylim(0, 3)
        plt.savefig("spikes.png")
        plt.show()

        return


    def gaussian(self, mu, sigma, x):
        """ Return sample from a gaussiam function (not random) """
        A = 1 / (np.sqrt(2 * np.pi * sigma * sigma))
        B = ((x - mu)**2) / (2 * sigma**2)
        return A * np.exp(-B)


    def generate_filters(self, mu_H, sigma_H, L):
        """
        Generate a tensor of finite impulse filters

        Filters are samples from Gaussian functions with different parameters
        Filter for the i_th channel j_th source is defined by parameters mu_H_ij and sigma_H_ij

        Args:
            mu_H (np.ndarray): Shape (m, n). Mean parameter of filters
            sigma_H (np.ndarray): Shape (m, n). std of filters
            L (int): (max) length of the finite impulse response filters in time steps

        Returns:
            np.ndarray: Shape (m, n, L). Tensor of filters.
        """

        m, n = self.num_channels, self.num_sources

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
        times = x * self.sampling_period

        # Sample filter from the gaussian using parameters
        return self.gaussian(mu_H, sigma_H, times)


    def visualise_filters(self, filters):
        """
        Visualise a number of filters
        """
        num_filters = filters.shape[0] * filters.shape[1]
        L = filters.shape[2]
        fig, axs = plt.subplots(num_filters, 1, figsize=(8, 8), sharey=True)

        x = np.arange(L) * self.sampling_period

        i = 0
        for c in filters:
            for s in c:
                axs[i].plot(x, s, "o", color="green")
                i += 1

        plt.show()
        return


    def lagged_spike_trains(self, spike_trains, L):
        """
        Transform a 2D tensor of spike trains into a 3D tensor of lagged spike trains

        This is to vectorise the convolution operation.

        Args:
            spike_trains (np.ndarray): Shape (n, T). Tensor of spike trains

        Returns:
            np.ndarray: Shape (n, L, T). Tensor of lagged spike trains. 
                S_{ijk} is impulse (0 or 1) of source i at time (k - j)
        """
        T = self.num_samples

        m, n = self.num_channels, self.num_sources

        t_indices = np.arange(T) 
        l_indices = np.arange(L)

        # Compute the shifted indices (broadcasting L over T)
        # Matrix of indeces with X_ij = j - i
        shifted_indices = t_indices[None, :] - l_indices[:, None]

        valid_mask = shifted_indices >= 0

        S = np.zeros((n, L, T))


        for j in range(n):  
            S[j] = np.where(valid_mask, spike_trains[j, shifted_indices], 0)

        return S
    
        # Was causing failed test cases
        """
        valid_mask = np.logical_and(shifted_indices >= 0, shifted_indices < T)
        for j in range(n):
            for l in range(L):
                for t in range(T):
                    if valid_mask[l, t]:
                        index = shifted_indices[l, t]
                        if index < 0 or index >= spike_trains.shape[1]:
                            raise IndexError(f"Index {index} out of bounds for spike_trains with shape {spike_trains.shape}")
                        value = spike_trains[j, index]
                        if not np.isscalar(value):
                            raise ValueError(f"Expected scalar at spike_trains[{j}, {index}], got {value}")
                        S[j, l, t] = value
            return S
        """


    def generate_data(self, firing_rates, filter_length, mu_H, sigma_H):
        """
        Generate a single datapoint

        Args:
            firing_rates (np.ndarray): Shape (n,). Firing rates of each source
            filter_length (int): Length of the filter in time steps. L in documentation.
            mu_H (np.ndarray): Shape (m, n). Mean parameter of filters
            sigma_H (np.ndarray): Shape (m, n). std of filters

        Returns: 
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (emg, spike_trains, H)
                - emg (np.ndarray): Shape (m, T). Simulated sEMG signals
                - spike_trains (np.ndarray): Shape (n, T). Spike trains
                - H (np.ndarray): Shape (m, n, L). Filters
        """
        m, n = self.num_channels, self.num_sources
        L, T = filter_length, self.num_samples

        spike_times = self.generate_spike_times_poisson(firing_rates=firing_rates)
        spike_trains = self.spike_times_to_trains(spike_times=spike_times)

        H = self.generate_filters(mu_H=mu_H, sigma_H=sigma_H, L=filter_length)

        # Convert tensors to vectorise convolution operation
        S = self.lagged_spike_trains(spike_trains=spike_trains, L=filter_length)
        H_prime = H.reshape(m, n*L)
        S_prime = S.reshape(n*L, T)

        emg = H_prime @ S_prime

        return (emg, spike_trains, H)


    def visualise_EMG(self, signals):
        """
        Visualise a number of EMG signals
        """
        fig, axs = plt.subplots(self.num_channels, 1, figsize=(8, 8), sharey=True)

        x = np.arange(self.num_samples) * self.sampling_period

        i = 0
        for channel in signals:
            axs[i].plot(x, channel, color="green")
            channel_times = channel * self.sampling_period
            axs[i].vlines(channel_times, ymin=0, ymax=1, color='green', linestyle='-')
            i += 1

        plt.savefig("EMG.png")
        plt.show()
        return


if __name__ == "__main__":


    gen = DataGenerator(channels=3, sources=2, duration=2, sampling_frequency=1000)

    firing_rates = np.random.randint(5, 30, (2))
    L = 30
    mu_H = np.array([[0.005, 0.01],
                    [0.015, 0.0075],
                    [0.01, 0.008]])


    sigma_H = np.array([[0.01, 0.015],
                        [0.01, 0.01], 
                        [0.01, 0.01]])

    X, Y, _ = gen.generate_data(firing_rates=firing_rates, filter_length=L, mu_H=mu_H, sigma_H=sigma_H)

    gen.visualise_spike_trains(Y)
    gen.visualise_EMG(X)


