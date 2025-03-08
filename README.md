# nafes


[![image](https://img.shields.io/pypi/v/nafes.svg)](https://pypi.python.org/pypi/nafes)
[![python: 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![github](https://img.shields.io/badge/version-0.0.4-yellow.svg)](https://github.com/naotoo1/nafes)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


**A python package for prototype-based feature selection**

Nafes is a prototype-based feature selection package designed as a wrapper centered on the highly interpretable and powerful Generalized Matrix Learning Vector Quantization (GMLVQ) classification algorithm and its local variant (LGMLVQ). Nafes utilizes the learned relevances evaluated by the mutation validation scheme for Learning Vector quantization (LVQ), which iteratively converges to selected features that relevantly contribute to the prototype-based classifier decisions. 

    

## Installation
Nafes can be installed using pip.
```python
pip install nafes
```

If you have installed Nafes before and want to upgrade to the latest version, you can run the following command in your terminal:
```python
pip install -U nafes
```

To install the latest development version directly from the GitHub repository:

```python
pip install git+https://github.com/naotoo1/nafes
```

## Development Environment
Nafes provides a fully reproducible development environment using Nix and [devenv](https://devenv.sh/getting-started/). Once you have installed Nix and devenv, you can do the following:

   ```bash
   mkdir -p ~/.config/nix
   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
   nix profile install --accept-flake-config "github:cachix/devenv/latest"
   ```

Then clone and enter the project directory:

```bash
git clone https://github.com/naotoo1/nafes.git
cd nafes
```

Activate the reproducible development environment:
   ```bash
devenv shell
   ```
You may optionally consider using [direnv](https://direnv.net/) for automatic shell activation when entering the project directory.

To install Nafes in development mode, follow these steps to set up your environment with all the necessary dependencies while ensuring the package is installed with live code editing capabilities. To use the local reproducible development enviroment, execute the following lock file commands:

```bash
# Generate requirements file
generate-requirements

# Update lock files
update-lock-files

# Install dependencies from lock file
install-from-lock
   ``` 
To use the reproducible docker container with support for GPU/CPU:

```bash
# For GPU support
create-reproducible-container
run-reproducible-container

# For CPU-only environments
create-cpu-container
run-cpu-container
   ```
When working with Nafes in development mode, changes to the code take effect immediately without reinstallation. Use ```git pull``` to get the latest updates from the repository. Run tests after making changes to verify functionality

## Bibtex
If you would like to cite the package, please use this:
```python
@misc{Otoo_nafes_2023,
author = {Otoo, Nana Abeka},
title = {nafes},
year = {2023},
publisher = {GitHub},
journal = {GitHub repository},
howpublished= {\url{https://github.com/naotoo1/nafes}},
}
```

