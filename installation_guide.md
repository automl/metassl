## Table of Contents

 * [Installation](#installation)
 * [General Usage](#general-usage)
    * [Activate environment](#activate-Environment)
    * [Just commands](#just-Commands)
    * [Do Not Run Pre-Commit Hooks](#do-not-run-pre-commit-hooks)


## Installation
Basic packages to install once (if not already installed yet):
```bash
bash install/miniconda.sh    # For managing python versions
bash install/poetry.sh       # Python package manager
bash install/just.sh         # Command runner
```
Create and activate your environment:
```bash
conda create -n metassl python=3.7  # NEPS currently supports only up to python 3.7
conda activate metassl
```
For automated formatting etc. for each commit:
```bash
poetry install  # Needs to be done before pre-commit install
pre-commit install
```

For further installations use the command:
```
poetry add <PACKAGE-TO-INSTALL>
```
Example: `poetry add matplotlib`

For more informations concerning poetry.lock and pyproject.toml see the poetry documentation for
[Dependency installation](https://python-poetry.org/docs/basic-usage/#installing-dependencies)

## General Usage

### Activate environment
```bash
conda activate metassl  # Always need to be activated when working!!!
```

### Just commands
List all just commands with the following command:
```bash
just list
```
Use a specific command via:
```bash
just <COMMAND NAME>
```

### Do not run pre-commit hooks

To commit without runnning `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.
