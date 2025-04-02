# Directory Breakdown
```
.
├── README.md
├── config
│   └── config.yaml
├── experiments
├── figures
├── notebooks
│   └── data_generation.ipynb
├── requirements.txt
├── scripts
│   ├── evaluate.py
│   └── train.py
├── setup.py
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── data_generation.py
│   │   ├── data_generator.py
│   │   ├── dataset.py
│   │   └── generate_data.py
│   ├── models
│   │   └── __init__.py
│   └── utils
│       └── __init__.py
└── tests
    ├── test_data.py
    └── test_models.py
```
## config/
Contains configuration files (e.g.,model, hyperparameters, data etc.) for each experiment

## experiments/
Holds data and results around a single experiment. This includes a copy of the config file used, logs, checkpoints, results and visual outputs (e.g. loss plot).

## figures/
Contains visualisations generated

## notebooks/
Stores Jupyter notebooks used for visualisation, prototyping and examples. 

## scripts/
Entry-point scripts for training, evaluation etc. 
These will usually require a config file to specify how to run. 

## src/
Contains core project code, such as for data processing and models.

## test/
Tests

__On local machine only__
## data/
Store dataset, with a number of processing checkpoints (e.g. raw/processed)

## models/
Stores serialised versions of trained models




## Example usage
TODO: Could include model-name in config file
Example (workflow) usage of how a user might train a model and get some results.

1. Setup python environment using requirements.txt
2. Create a config file specifying model type, hyperparemeters etc.
3. Train a model using this config
	`python -m scripts.train --config configs/config.yaml`
4. Model will be trained, with logs, checkpoints and visualisations saved to `experiments/model-name/train`. Model is then serialised and saved in `models/` with directory name specified in config file (same as `model-name`). The `config.yaml` file is also be copied into this folder.
5. Test the model
	`python scripts/evaluate.py models/model-name`
6. Results and any visualisations will be saved to `experiments/model-name/evaluate`
7. Use for inference
	`python scripts/predict.py models/model-name data-path`

__Alternative__

3. Train and test model using a config
	`python scripts/train-test.py config.yaml`
4. Model will be trained, with logs, checkpoints, visualisations and results saved to `experiments/model-name/`. Model is then serialised and saved in `models/` with directory name specified in config file (same as `model-name`). The `config.yaml` file is also be copied into the experiments and models folders.