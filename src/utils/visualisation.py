"""
Module visualisation

Contains methods for visualisation of EMG and related data

Usage:
    from src.utils.visualisation import visualise_EMG

    visualise_EMG(data)

"""

import matplotlib.pyplot as plt
import numpy as np


def visualise_spike_trains(spike_trains):
    """
    Visualise a number of spike trains

    Args:
        spike_trains (torch.Tensor): Shape (T,).
            NOTE this is different dtype to visualise functions within data_generator
    """

    num_sources, num_samples = spike_trains.shape[0], spike_trains.shape[1]

    fig, axs = plt.subplots(num_sources, 1, figsize=(8, 8), sharey=True)

    x = np.arange(num_samples)

    i = 0
    for source in spike_trains:
        axs[i].plot(x, source, 'o', color="blue")
        spike_times = np.nonzero(source)
        axs[i].vlines(spike_times, ymin=0, ymax=1, color='blue', linestyle='-')
        i += 1

    plt.ylim(0, 3)
    plt.show()
    return


def visualise_filters(filters):
    """
    Visualise a number of filters
    """
    num_filters = filters.shape[0] * filters.shape[1]
    L = filters.shape[2]
    fig, axs = plt.subplots(num_filters, 1, figsize=(8, 8), sharey=True)

    x = np.arange(L)

    i = 0
    for c in filters:
        for s in c:
            axs[i].plot(x, s, "o", color="green")
            i += 1

    plt.show()
    return


def visualise_EMG(signals):
    """
    Visualise a number of EMG signals
    """
    num_channels, num_samples = signals.shape[0], signals.shape[1]

    fig, axs = plt.subplots(num_channels, 1, figsize=(8, 8), sharey=True)

    x = np.arange(num_samples)

    i = 0
    for channel in signals:
        axs[i].plot(x, channel, color="green")
        channel_times = channel
        axs[i].vlines(channel_times, ymin=0, ymax=1, color='green', linestyle='-')
        i += 1

    plt.show()
    return