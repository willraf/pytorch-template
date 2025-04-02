import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

def create_experiment_directory(config):
    """Create a directory for the experiment with a structured naming convention."""
    model_name = config['model']['name']
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    experiment_name = f"{model_name}_{timestamp}"
    experiment_dir = os.path.join("experiments", experiment_name)

    # Create directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "figures"), exist_ok=True)

    return experiment_dir

def setup_logger(experiment_dir): 
    """
    Setup logger for experiment

    TODO:
        - Use separate loggers for training and evaluation
    """
    log_dir = os.path.join(experiment_dir, "logs")
    # This line should be redundant
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "experiment.log")),
            logging.StreamHandler()
        ]
    )

def change_log_file(experiment_dir, new_log_file):
    
    log_dir = os.path.join(experiment_dir, "logs")
    # This line should be redundant
    os.makedirs(log_dir, exist_ok=True)

    log_fpath = os.path.join(log_dir, new_log_file)

    logger = logging.getLogger()  # Get the root logger
    
    # Remove existing file handlers
    for handler in logger.handlers[:]:  
        if isinstance(handler, logging.FileHandler):  
            logger.removeHandler(handler)
            handler.close()

    # Add new file handler
    file_handler = logging.FileHandler(log_fpath)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Log file changed to: {new_log_file}")

def plot_and_save_loss_curves(results, experiment_dir):
    """
    Plot training and validation loss curves and save the plot to a file.

    Args:
        results (dict): Dictionary containing epoch, train_loss, and val_loss.
        experiment_dir (str): Path to the experiment directory.
    """
    # Create the figures directory if it doesn't exist
    figures_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(results["epochs"], results["train_loss"], label="Training Loss", color="blue")
    plt.plot(results["epochs"], results["val_loss"], label="Validation Loss", color="orange")

    # Add labels, title, and legend
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Save the plot to a file
    plot_path = os.path.join(figures_dir, "train.png")
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Loss curves saved to {plot_path}")
