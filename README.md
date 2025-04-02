# EMG Signal Separation
Date created: 26/11/2024

Brief description of your project.

## TODO
- Edit main.py to work using config files to define all hyperparameters and variables (e.g. for model and for data generation)
- Automatically save experiment results and model snapshots
- Write tests for data_generator.py

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

## Git workflow 

1. `git pull`

2. Make changes

3. `git add file_name`

4. `git commit -m "commit message"`

5. `git pull` (again)

6. Fix any conflicts

7. `git push`