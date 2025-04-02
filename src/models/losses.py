import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


""" Module losses

Contains Losses class which contains functions for computing losses.
Test code exists in notebooks/losses_tests.ipynb

Usage:

Authors:

Date: 30/1/2025

TODO:
    - Think about how sparsity loss works, because we don't want to have no spikes at all
    - Hunagarian loss
    - Add other kinds of reduction to hungarian_loss

Notes:
    - Added reduction parameter to hungarian_loss. 
        Think this might be irrelavent atm as loss is computed over each individual datapoint and 
        then averaged over the batch.
"""

class Losses:
    """
    Contains different loss functions including
    - positional_loss: Compares the model output with the ground truth spike train
    - sparsity: Compute the negative negentropy loss term to promote sparseness in a predicted spike train
    - distance_penalty: Compute the penalty for violating the minimum distance between consecutive spikes
    - num_spikes_penalty: Compute the penalty for exceeding the maximum allowed number of spikes
    """

    def __init__(self):
        pass

    def positional_loss(self, spike_train_output, spike_train_input, kernel_size=7, sigma=1.5, visualising_convolved=False):
        """
        Compares one spike train with another, using Gaussian convolution to allow for small timing differences.
        i.e. one row compared to another row.
        
        Args:
            spike_train_output (torch.Tensor): Single output spike train (T,)
            spike_train_input (torch.Tensor): Single input spike train (T,)
            kernel_size (int): Length of the Gaussian kernel. It must be odd!
            sigma (float): Standard deviation of the Gaussian kernel
            visualising_convolved (bool): If True, return the convolved input as well as the positional loss
        Returns:
            torch.Tensor: The positional loss
            torch.Tensor: The convolved input (so that it can be plotted in losses_tests.ipynb)
        """
        # Get the device of the input tensors
        device = spike_train_output.device

        # Ensure 1D tensors
        if spike_train_output.dim() > 1:
            spike_train_output = spike_train_output.squeeze()
        if spike_train_input.dim() > 1:
            spike_train_input = spike_train_input.squeeze()

        # Generate 1D gaussian kernel
        # return 1D tensor of size kernel_size centred at 0
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
        gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max()  # Normalize so peak of gaussian is 1
        gaussian_kernel = gaussian_kernel.view(1, 1, -1) # Reshape to (1, 1, kernel_size) for compatibility with conv1d
        gaussian_kernel = gaussian_kernel.to(device)

        # Convolution requires (batch size, channels, length)
        compatible_input = spike_train_input.unsqueeze(0).unsqueeze(0) # (1, 1, T)
        
        # Convolve the source with the kernel
        padding = (kernel_size - 1) // 2
        convolved = F.conv1d(compatible_input, gaussian_kernel, padding=padding) # (1, 1, T)
        convolved_input = convolved.squeeze(0).squeeze(0)  # Back to 1D so (T,)
        
        # Ensure the convolved input has the same length as the original input
        if convolved_input.size(0) != spike_train_input.size(0):
            convolved_input = convolved_input[:spike_train_input.size(0)]

        # Compute the positional loss
        loss = F.mse_loss(spike_train_output, convolved_input)

        if visualising_convolved:
            return loss, convolved_input
        else:
            return loss

    def sparsity_loss(self, spike_train_output, bin_width=0.1):
        """
        Compute the negative negentropy loss term to promote sparseness in a predicted spike train.

        Args:
            spike_train_output (torch.Tensor): A 1D PyTorch tensor representing the predicted spike train.
            bin_width (float): The width of the bins used to compute the histogram for the entropy calculation.

        Returns:
            torch.Tensor: The negative negentropy loss term.
        """
        device = spike_train_output.device

        # Ensure the spike train is a 1D PyTorch tensor
        spike_train = spike_train_output.flatten()

        # Compute the entropy of the spike train
        # Approximate probability distribution using a histogram
        range_data = spike_train.max() - spike_train.min()  # Calculate the range of the data
        num_bins = int(range_data / bin_width)  # Calculate the number of bins based on bin width
        num_bins = max(num_bins, 1) # Ensure num_bins is at least 1
        hist = torch.histc(spike_train, bins=num_bins, min=spike_train.min(), max=spike_train.max())
        hist = hist.to(device)
        hist /= hist.sum() # Normalize histogram to get probability distribution

        # Compute the entropy using the formula: H(X) = -sum(p(x) * log(p(x)))
        # Ignore zero values in the histogram
        # This is because 0 * log(0) is defined to be 0, and should not contribute to entropy.
        entropy_train = -(hist[hist > 0] * torch.log(hist[hist > 0])).sum()

        # Entropy of a Gaussian distribution with same mean and variance
        mean_train = spike_train.mean()
        var_train = spike_train.var()

        # Entropy of a Gaussian distribution with mean and variance of spike train
        entropy_gaussian = 0.5 * torch.log(2 * torch.pi * torch.e * var_train+ 1e-8)  # Added small constant for numerical stability

        # Compute the negentropy (measure of non-gaussianity)
        negentropy = entropy_gaussian - entropy_train

        # So the loss term we wnat to minimise is the negative negentropy
        negative_negentropy_loss = -negentropy

        return negative_negentropy_loss


    # Upper and lower limit constraint penalties (Lagrange multipliers)
    def distance_penalty(self, spike_train_output, D_min):
        """
        Compute the penalty for violating the minimum distance between consecutive spikes

        Args:
            spike_train_output (torch.Tensor): A 1D PyTorch tensor representing the predicted spike train.
            D_min (float): The minimum allowed distance between spikes.

        Returns:
            torch.Tensor: The penalty for violating the minimum distance constraint.
        """
        device = spike_train_output.device
        # Find indices where spikes occur (value = 1)
        spike_indices = torch.nonzero(spike_train_output == 1).flatten()

        # If there are less than 2 spikes, no penalty for distance
        if len(spike_indices) <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Calculate distances between consecutive spikes
            distances = spike_indices[1:] - spike_indices[:-1]
            D_min = torch.tensor(D_min, device=device)

            # Compute the penalty if any pair of consecutive spikes is closer than D_min
            penalty = torch.sum(torch.max(torch.tensor(0.0, device=device), D_min - distances.float()))

            return penalty
        
    def num_spikes_penalty(self, spike_train_output, N_max):
        """
        Compute the penalty for exceeding the maximum allowed number of spikes.

        Args:
            spike_train_output (torch.Tensor): A 1D PyTorch tensor representing the predicted spike train.
            N_max (int): The maximum allowed number of spikes.

        Returns:
            torch.Tensor: The penalty for violating the number of spikes constraint.
        """
        device = spike_train_output.device

        # Count the number of spikes
        num_spikes = torch.sum(spike_train_output == 1)
        N_max = torch.tensor(N_max, device=device)

        # Compute the penalty if the number of spikes exceeds N_max
        penalty = torch.max(torch.tensor(0.0, device=device), num_spikes - N_max)
        
        return penalty

    def total_loss(self, model_output, input_trains, sparsity_weight=0.1, distance_weight=0.1, num_spikes_weight=0.1):
        """
        Computes the total loss as a weighted sum of the positional loss, sparsity loss, distance penalty, and number of spikes penalty.

        Args:
            model_output (torch.Tensor): The output of the model.
            input_trains (torch.Tensor): The ground truth spike trains.
            sparsity_weight (float): The weight for the sparsity loss.
            distance_weight (float): The weight for the distance penalty.
            num_spikes_weight (float): The weight for the number of spikes penalty.

        Returns:
            torch.Tensor: The total loss.
        """
        # Compute the positional loss
        positional_loss = self.positional_loss(model_output, input_trains)

        # Compute the sparsity loss
        sparsity_loss = self.sparsity_loss(model_output)

        # Compute the distance penalty
        distance_penalty = self.distance_penalty(model_output, D_min=1)

        # Compute the number of spikes penalty
        num_spikes_penalty = self.num_spikes_penalty(model_output, N_max=10)

        # Compute the total loss as a weighted sum of the individual losses
        total_loss = positional_loss + sparsity_weight * sparsity_loss + distance_weight * distance_penalty + num_spikes_weight * num_spikes_penalty

        return total_loss
    
    def hungarian_loss_wrapper(self, loss_fn):
        """
        Wrapper for the hungarian loss function for code consistency.

        Returns a function that computes the hungarian loss using the specified loss function.

        Args:
            loss_fn (callable): A function that computes the loss between two tensors.
                                This should be a row-wise loss function like MSE or cross-entropy.
        
        Returns:
            callable: A function that takes (outputs, targets) and returns the Hungarian loss.
        """
        def wrapped(outputs, targets, reduction='mean'):
            return self.hungarian_loss(outputs, targets, loss_fn=loss_fn, reduction=reduction)  
        
        return wrapped
    
    def hungarian_loss(self, model_output, input_trains, loss_fn, reduction='mean'):
        """
        Computes permutation-invariant loss using Hungarian matching.
        This is expensive for large sets, but for fewer number of channels this should be good.
        60 electrodes/channels can in a good case decompose 20 sources
        Handles NaN and Inf values in cost matrix by falling back to direct assignment.
        
        Args:
            model_output: (sources, num_timebins) - Predictions
            input_trains: (sources, num_timebins) - Ground truth
            loss_fn: Function to compute row-wise loss (this should be the positional loss)
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'

        Returns:
            Permutation-invariant loss
        """
        batch_size, num_rows, num_timebins = input_trains.shape
        device = model_output.device
        losses = []

        for i in range(batch_size):
            # Compute pairwise loss matrix (num_rows x num_rows)
            cost_matrix = torch.zeros((num_rows, num_rows), device=device)

            for j in range(num_rows):  # Iterate over each row in input_trains
                for k in range(num_rows):  # Iterate over each row in model_output
                    # Ensure both inputs are 1D tensors and pass to positional_loss
                    input_spike_train = input_trains[i, j].flatten()  # Flatten to 1D if not already
                    model_output_spike_train = model_output[i, k].flatten()  # Flatten to 1D if not already
                    
                    # Compute loss between input_trains[j] and model_output[k]
                    cost = loss_fn(input_spike_train, model_output_spike_train)
                    cost_val = cost.item() if isinstance(cost, torch.Tensor) else cost
                    
                    # Handle NaN or Inf values
                    if np.isnan(cost_val) or np.isinf(cost_val):
                        cost_matrix[j, k] = 1e6  # Large but finite value
                    else:
                        cost_matrix[j, k] = cost_val

                    # # Compute loss between input_trains[j] and model_output[k]
                    # cost_matrix[j, k] = loss_fn(input_spike_train, model_output_spike_train)

            # Convert to NumPy for Hungarian algorithm
            cost_matrix_np = cost_matrix.detach().cpu().numpy()

            # Solve optimal row assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            # Convert indices to tensors on the correct device
            row_ind = torch.tensor(row_ind, device=device)
            col_ind = torch.tensor(col_ind, device=device)
            
            # # Print the matched rows for this batch
            # print(f"Batch {i + 1}: Matched rows:")
            # for r, c in zip(row_ind, col_ind):
            #     print(f"  - Input row {r} matched with Model output row {c}")

            # Compute total loss using optimally matched rows
            matched_loss = loss_fn(input_trains[i, row_ind].flatten(), model_output[i, col_ind].flatten())
            losses.append(matched_loss)

        # sum all the losses and divide by batch size
        return torch.stack(losses).mean()

if __name__ == "__main__":
    pass

        

