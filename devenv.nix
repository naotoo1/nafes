{ pkgs, lib, config, inputs, ... }:
let
  pythonPackages = pkgs.python312Packages;
  project_dir = "${config.env.DEVENV_ROOT}";
  env.PYTHONPATH = "${pkgs.python312Packages.sitePackages}";
in {
  cachix.enable = false;
  env.PROJECT_DIR = project_dir;
  packages = with pkgs; [
    git
    git-lfs
    nixpkgs-fmt
    pythonPackages.numpy
    pythonPackages.matplotlib
    pythonPackages.scikit-learn
    pythonPackages.pandas
    pythonPackages.scipy
    pythonPackages.torch
    pythonPackages.pytorch-lightning
    pythonPackages.pip
  ];
  
  languages = {
    python = {
      enable = true;
      package = pythonPackages.python;
      uv = {
        enable = true;
      };
      venv = {
        enable = true;
      };
    };
  };
  
  # Scripts for managing non-nix packages with UV
  scripts = {
    update-non-nix-packages.exec = ''
      echo "Creating requirements.in with non-nix-packages..."
      echo "prototorch" > requirements.in
      echo "prototorch_models" >> requirements.in
      echo "Generating uv.lock file..."
      uv pip compile requirements.in -o uv.lock
      echo "uv.lock generated with non-nix packages."
    '';
    
    install-from-lock.exec = ''
      echo "Installing Python packages from uv.lock..."
      if [ -f uv.lock ]; then
        uv pip sync uv.lock
        echo "Packages installed successfully from uv.lock."
      else
        echo "uv.lock not found. Run update-prototorch first."
      fi
    '';
    
    quick-install-non-nix-packages.exec = ''
      echo "Quick installing prototorch packages with uv..."
      uv pip install prototorch prototorch_models
      echo "Prototorch packages installed. Note: This doesn't update uv.lock."
    '';
  };
  
  starship.enable = true;
  
  # Run on first entry to the environment
  enterShell = ''
    echo "Python environment activated."
    echo "- To generate a lock file for non-nix-packages: run 'update-non-nix-packages.exec'"
    echo "- To install packages from lock file: run 'install-from-lock'"
    echo "- For quick installation without lock: run 'quick-install-non-nix-packages.exec'"
  '';
}