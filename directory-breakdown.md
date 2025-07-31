# Directory Breakdown
`git ls-files | tree --fromfile`

```
.
├── configs
│   ├── default.yaml
│   └── main.yaml
├── directory-breakdown.md
├── .gitignore
├── notebooks
│   └── example_notebook.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── eval.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── setup.py
├── src
│   ├── data
│   │   ├── base_dataset.py
│   │   ├── curriculum_scheduler.py
│   │   ├── example_dataset.py
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── __init__.py
│   ├── models
│   │   ├── example_model.py
│   │   ├── __init__.py
│   │   └── setup.py
│   └── utils
│       ├── config_wrapper.py
│       ├── __init__.py
│       ├── utils.py
│       └── visualisation.py
└── tests
    └── example_test.py

```
## config/
Contains configuration files (e.g.,model, hyperparameters, data etc.) for each experiment.
- main.yaml: Parameters for full pipeline. These can optionally be split into train.yaml, eval.yaml etc.
- default.yaml: Defines default params. Any parameters redifined in main.yaml will override these defaults. Parameters left null here are required to be set in main.yaml.

## experiments/
Holds data and results around a single experiment. This includes a copy of the config file used, logs, checkpoints, results and visual outputs (e.g. loss plot).

## figures/
Contains general visualisations generated

## notebooks/
Stores Jupyter notebooks used for visualisation, prototyping and examples. 

## scripts/
Entry-point scripts for training, evaluation etc. 
These will usually require a config file to specify how to run. 

## src/
Contains core project code, such as for data processing and models.

### src/data/
__init__.py exposes the setup_data function in setup.py. This automatically instantiates a dataset from config file and returns dataloaders

### src/models/
Similar to data but with setup_model funciton

### src/utils/
- config_wrapper.py: Contains a class to wrap the config file for easier retreval and default setting. 
- utils.py: Contains useful functions used by train, eval and/or predict

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
	`python -m scripts.train --config configs/train.yaml`
4. Model will be trained, with logs, checkpoints and visualisations saved to `experiments/experiment-name/train`. The `config.yaml` file is also be copied into this folder.
5. Test the model
	`python scripts/evaluate.py --config configs/eval.yaml`
6. Results and any visualisations will be saved to `experiments/experiment-name/evaluate`
7. Use for inference
	`python scripts/predict.py --config configs/predict.yaml`

__Alternative__

3. Run full pipeline using a config
	`python scripts/train-test.py config.yaml`
4. Model will be trained, with logs, checkpoints, visualisations and results saved to `experiments/model-name/` The `config.yaml` file is also be copied into the experiments and models folders..