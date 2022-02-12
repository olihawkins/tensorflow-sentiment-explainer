# Environment

This file documents the steps taken to create and recreate the environment. The environment was originally created with `venv` using a Python installation managed by Homebrew on MacOS. But the same packages could equally be installed into an environment managed by conda or another Python distribution.

## Create the environment

```zsh
/usr/local/bin/python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

## Install packages

```zsh
pip install ipython numpy pandas scikit-learn matplotlib tensorflow tensorflow-datasets shap
```

## Activate the environment

```zsh
source env/bin/activate
```

## Deactivate the environment

```zsh
source env/bin/deactivate
```

## VS Code integration

If using VS Code set the project interpreter to: `./env/bin/python`