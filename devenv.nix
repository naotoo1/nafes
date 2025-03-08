{ pkgs, lib, config, inputs, ... }:
let
  pythonPackages = pkgs.python312Packages;
  project_dir = "${config.env.DEVENV_ROOT}";
in {
  cachix.enable = false;
  
  env = {
    PROJECT_DIR = project_dir;

    # Ensure Nix packages are discoverable by Python
    PYTHONPATH = lib.makeSearchPath "lib/python3.12/site-packages" [
      pythonPackages.numpy
      pythonPackages.matplotlib
      pythonPackages.scikit-learn
      pythonPackages.pandas
      pythonPackages.scipy
      pythonPackages.torch
      pythonPackages.torchvision
      pythonPackages.pytorch-lightning
    ];
  };

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
    pythonPackages.torchvision
    pythonPackages.pytorch-lightning
    pythonPackages.pip
    docker
    docker-compose
    nix-prefetch-git
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

  scripts = {
    # Generates requirements.in without Nix packages
    generate-requirements.exec = ''
      echo "Generating requirements.in without Nix packages..."
      cat > requirements.in << EOL
# Non-Nix packages
prototorch
prototorch_models
-e .
EOL
      echo "requirements.in generated."
    '';

    # Updates uv.lock using requirements.in
    update-lock-files.exec = ''
      echo "Updating lock files..."
      
      # Generate requirements.in
      ./generate-requirements

      # Generate uv.lock
      echo "Generating uv.lock file..."
      uv pip compile requirements.in -o uv.lock

      echo "Lock files updated successfully."
    '';

    # Installs non-Nix Python packages from uv.lock
    install-from-lock.exec = ''
      echo "Installing Python packages from uv.lock..."
      if [ -f uv.lock ]; then
        uv pip sync uv.lock
        echo "Non-Nix packages installed successfully."
      else
        echo "uv.lock not found. Run update-lock-files first."
      fi
    '';

    quick-install-non-nix-packages.exec = ''
      echo "Quick installing non-Nix packages with uv..."
      uv pip install prototorch prototorch_models -e .
      echo "Prototorch packages installed."
    '';

    # Docker container with lock files for reproducibility
    create-reproducible-container.exec = ''
      echo "Creating reproducible GPU-enabled container from lock files..."
      
      # Ensure lock files exist
      if [ ! -f uv.lock ]; then
        echo "uv.lock not found. Generating requirements.in and uv.lock now..."
        $PROJECT_DIR/generate-requirements
        uv pip compile requirements.in -o uv.lock
      fi
      
      # Create a consolidated requirements.txt from uv.lock - excluding the local package
      echo "Generating frozen requirements from uv.lock without local package..."
      
      # Create a modified requirements.in without the local package for Docker
      grep -v "^-e \." requirements.in > requirements-docker.in
      
      # Create frozen requirements from the Docker-specific input
      uv pip compile requirements-docker.in -o requirements-frozen-docker.txt
      
      # Create Dockerfile that uses the lock files
      cat > Dockerfile << EOF
      FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04
      
      # System dependencies
      RUN apt-get update && apt-get install -y \\
          python3 \\
          python3-pip \\
          python3-dev \\
          git \\
          git-lfs \\
          && rm -rf /var/lib/apt/lists/*
      
      # Create working directory
      WORKDIR /app
      
      # Upgrade pip and setuptools first
      RUN pip3 install --no-cache-dir --upgrade pip setuptools>=61
      
      # Copy the entire project
      COPY . /app
      
      # Copy Docker-specific frozen requirements
      COPY requirements-frozen-docker.txt /app/requirements-frozen.txt
      
      # Install dependencies from frozen requirements
      RUN pip3 install --no-cache-dir -r requirements-frozen.txt
      
      # Install the local package (not in editable mode)
      RUN pip3 install --no-cache-dir -e .
      
      # Set environment variables for PyTorch
      ENV PYTHONPATH=/usr/local/lib/python3/dist-packages
      
      # Default command
      CMD ["python3"]
      EOF
      
      echo "Building Docker image from frozen requirements..."
      docker build -t gpu-prototorch-env-locked .
      echo "Reproducible image built successfully."
    '';

    run-reproducible-container.exec = ''
      echo "Running reproducible GPU container..."
      docker run --gpus all -it --rm \
        -v "$(pwd):/app" \
        gpu-prototorch-env-locked bash
    '';

    # CPU-only container for systems without GPUs
    create-cpu-container.exec = ''
      echo "Creating CPU-only container from lock files..."
      
      # Ensure uv.lock exists
      if [ ! -f uv.lock ]; then
        echo "uv.lock not found. Generating requirements.in and uv.lock now..."
        $PROJECT_DIR/generate-requirements
        uv pip compile requirements.in -o uv.lock
      fi
      
      # Use the same approach as GPU container - create requirements without local package
      if [ ! -f requirements-docker.in ]; then
        grep -v "^-e \." requirements.in > requirements-docker.in
      fi
      
      # Create frozen requirements from the Docker-specific input
      uv pip compile requirements-docker.in -o requirements-frozen-docker.txt
      
      # Create CPU Dockerfile
      cat > Dockerfile.cpu << EOF
      FROM python:3.12-slim
      
      # System dependencies
      RUN apt-get update && apt-get install -y \\
          git \\
          git-lfs \\
          && rm -rf /var/lib/apt/lists/*
      
      # Create working directory
      WORKDIR /app
      
      # Upgrade pip and setuptools first
      RUN pip3 install --no-cache-dir --upgrade pip setuptools>=61
      
      # Copy the entire project
      COPY . /app
      
      # Copy Docker-specific frozen requirements
      COPY requirements-frozen-docker.txt /app/requirements-frozen.txt
      
      # Install dependencies from frozen requirements
      RUN pip3 install --no-cache-dir -r requirements-frozen.txt
      
      # Install the local package (not in editable mode)
      RUN pip3 install --no-cache-dir -e .
      
      # Default command
      CMD ["python3"]
      EOF
      
      echo "Building CPU Docker image from frozen requirements..."
      docker build -t cpu-prototorch-env-locked -f Dockerfile.cpu .
      echo "CPU image built successfully."
    '';

    run-cpu-container.exec = ''
      echo "Running CPU-only container..."
      docker run -it --rm \
        -v "$(pwd):/app" \
        cpu-prototorch-env-locked bash
    '';   

  };

  starship.enable = true;
  
  enterShell = ''
    echo "=== Python Environment Setup ==="
    echo ""
    echo "Lock file commands:"
    echo "- Generate requirements.in:  run 'generate-requirements'"
    echo "- Update lock files:         run 'update-lock-files'"
    echo "- Install from lock:         run 'install-from-lock'"
    echo ""
    echo "Docker commands:"
    echo "- Create GPU container:      run 'create-reproducible-container'"
    echo "- Run GPU container:         run 'run-reproducible-container'"
    echo "- Create CPU container:      run 'create-cpu-container'"
    echo "- Run CPU container:         run 'run-cpu-container'"
  '';
}