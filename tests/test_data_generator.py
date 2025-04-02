import unittest
import numpy as np
from src.data.data_generator import DataGenerator

class TestDataGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up a DataGenerator instance for testing."""
        self.channels = 3
        self.sources = 2
        self.duration = 2.0
        self.sampling_frequency = 1000
        self.filter_length = 30  # time steps

        
        self.gen = DataGenerator(
            sources=self.sources,
            channels=self.channels,
            duration=self.duration,
            sampling_frequency=self.sampling_frequency
        )
        
        self.firing_rates = np.random.randint(5, 30, (self.sources,))
        self.mu_H = np.random.uniform(0.005, 0.015, (self.channels, self.sources))
        self.sigma_H = np.random.uniform(0.005, 0.015, (self.channels, self.sources))

    def test_generate_spike_times_poisson(self):
        """Test if spike times are generated correctly."""
        spike_times = self.gen.generate_spike_times_poisson(self.firing_rates)
        self.assertIsInstance(spike_times, np.ndarray)
        self.assertEqual(spike_times.shape[1], self.sources)  # Check number of sources
        self.assertTrue(np.all(spike_times[~np.isnan(spike_times)] >= 0))   # Ensure non-negative times
        # Ensure spike times are within the simulation duration
        self.assertTrue(np.all(spike_times[~np.isnan(spike_times)] <= self.duration))

        # Check whether the average number of spikes is close to the expected value
        trials = 100
        cummulative_spikes = np.zeros(self.sources)
        for _ in range(trials):
            spike_times = self.gen.generate_spike_times_poisson(self.firing_rates)
            cummulative_spikes += np.sum(~np.isnan(spike_times), axis=0)

        avg_spikes = cummulative_spikes / trials
        avg_firing_rates = avg_spikes / self.duration

        # Assert that the average number of spikes is close to the expected value with a tolerance of 10%
        self.assertTrue(np.allclose(avg_firing_rates, self.firing_rates, rtol=0.1))

    def test_spike_times_to_trains(self):
        """Test if spike times are correctly converted to spike trains."""
        spike_times = self.gen.generate_spike_times_poisson(self.firing_rates)
        spike_trains = self.gen.spike_times_to_trains(spike_times)

        self.assertIsInstance(spike_trains, np.ndarray)
        self.assertEqual(spike_trains.shape, (self.sources, self.gen.num_samples))

        # Ensure spike trains are binary (0 or 1)
        self.assertTrue(np.all(np.logical_or(spike_trains == 0, spike_trains == 1)))

    def test_generate_filters(self):
        """Test if filters are generated correctly."""
        filters = self.gen.generate_filters(self.mu_H, self.sigma_H, self.filter_length)

        self.assertIsInstance(filters, np.ndarray)
        self.assertEqual(filters.shape, (self.channels, self.sources, self.filter_length))

        # Ensure filters are non-negative (Gaussian filters)
        self.assertTrue(np.all(filters >= 0))

    def test_lagged_spike_trains(self):
        """Test if lagged spike trains are generated correctly."""
        spike_times = self.gen.generate_spike_times_poisson(self.firing_rates)
        spike_trains = self.gen.spike_times_to_trains(spike_times)
        lagged_spike_trains = self.gen.lagged_spike_trains(spike_trains, self.filter_length)

        self.assertIsInstance(lagged_spike_trains, np.ndarray)
        self.assertEqual(lagged_spike_trains.shape, (self.sources, self.filter_length, self.gen.num_samples))

        # Check property given in docs as $S_{j,l,t} = s_{j,t-l}$
        for n in range(self.sources):
            for l in range(self.filter_length):
                for t in range(self.gen.num_samples):
                    if t - l >= 0:
                        self.assertEqual(lagged_spike_trains[n, l, t], spike_trains[n, t - l])
                    else:
                        self.assertEqual(lagged_spike_trains[n, l, t], 0)


    # def test_lagged_spike_trains(self):
    #     """Test if lagged spike trains are generated correctly."""
    #     spike_trains = np.array([[1, 1, 1, 0, 0, 1],
    #                              [0, 1, 0, 0, 0, 0]])  

    #     true_lagged_spike_trains = np.array([[[1, 0], [0, 0]],
    #                                           [[1, 1], [1, 0]],
    #                                           [[1, 1], [0, 1]],
    #                                           [[0, 1], [0, 0]],
    #                                           [[0, 0], [0, 0]],
    #                                           [[1, 0], [0, 0]]]) 

    #     true_lagged_spike_trains = true_lagged_spike_trains.transpose(1, 2, 0)
    #     lagged_spike_trains = self.gen.lagged_spike_trains(spike_trains, 2, T=6)
    #     self.assertTrue(np.all(lagged_spike_trains == true_lagged_spike_trains))


    def test_generate_data(self):
        """Test if data generation works as expected."""
        emg, spike_trains, filters = self.gen.generate_data(
            firing_rates=self.firing_rates,
            filter_length=self.filter_length,
            mu_H=self.mu_H,
            sigma_H=self.sigma_H
        )

        self.assertIsInstance(emg, np.ndarray)
        self.assertIsInstance(spike_trains, np.ndarray)
        self.assertIsInstance(filters, np.ndarray)

        self.assertEqual(emg.shape, (self.channels, self.gen.num_samples))
        self.assertEqual(spike_trains.shape, (self.sources, self.gen.num_samples))
        self.assertEqual(filters.shape, (self.channels, self.sources, self.filter_length))


if __name__ == "__main__":
    unittest.main()
