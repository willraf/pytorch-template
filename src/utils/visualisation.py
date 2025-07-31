"""
Module visualisation

Contains methods for visualisation of EMG and related data

Usage:
    import src.utils.visualisation as vis
    vis.plot_emg(emg_data, fpath="path/to/save/plot.png")

Author:
    Will Raftery   

Date:
    31/07/2025

"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional

import torch 


def _convert_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch.Tensor to numpy array if needed."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def _save_or_show(fig: plt.Figure, fpath: Union[str, Path] = None) -> None:
    """Save figure to file if path provided, otherwise show it."""
    if fpath:
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath)
        logging.info(f"Saved plot to {fpath}")
    else:
        plt.show()
    plt.close(fig)


# EMG Signal Visualization
def plot_emg(emg: Union[np.ndarray, torch.Tensor], fpath: Optional[str] = None) -> None:
    """Plot EMG signals from multiple channels.
    
    Args:
        emg: EMG signals of shape (num_channels, time_steps)
        fpath: Optional path to save the plot
    """
    emg = _convert_to_numpy(emg)
    num_channels, time_steps = emg.shape
    
    fig, ax = plt.subplots(figsize=(5, 5))
    time = np.arange(time_steps)
    
    for ch in range(num_channels):
        ax.plot(time, emg[ch] + ch * 5, label=f'Channel {ch+1}')
    
    ax.set_title('EMG Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude + Offset')
    ax.grid(False)
    
    _save_or_show(fig, fpath)

