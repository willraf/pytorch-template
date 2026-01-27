# PyTorch Project Template
Date created: 30/07/2025

A example template of a pytorch project to reduce writing boilerplate code. 
Uses YAML config files

Check out directory-breakdown.md for in depth view of each file

## TODO 
- Add simple tests for each part

## Example Workflow

1. Fork the repo, then clone your fork locally
2. Setup a new virtual environment
3. Create a dataset in "src/datasets/[dataset_name]_dataset.py". The class [DatasetName]Dataset must exist inside. Check dataloader logic in "src/datasets/setup.py"
4. Create a model in "src/datasets/[model_name]_dataset.py". The class [ModelName]Model must exist inside. Check logic in "src/model/setup.py" works
5. In "src/utils/utils.py", finish implementing setup_loss_function() and setup_optimiser()
6. Add to config files as you go
7. Debug

## Virtual Environment Setup

__Python version 3.12__
Pytorch not yet available on 3.13

1. Create virtual environment named "venv"

`$ python3 -m venv venv`

2. If this has been created in this folder, add the name to the .gitignore file

3. Activate the virtual environment

`$ source venv`

4. Install dependancies from requirements.txt

`pip install -r requirements.txt`

### If using conda

1. `conda create --name bss312 python=3.12`

2. `conda activate bss312`

3. `conda install pip`

4. `pip install -r requirements.txt`

If changing environment dependamcies (e.g. with `pip install numpy`):

`pip freeze > requirements.txt`
`git add requirements.txt`


## Resources used
[PyTorch Framework](https://github.com/branislav1991/PyTorchProjectFramework/tree/master)

[Config files and wrappers](https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957)
