# 6.5930 Course Project

This repository contains notebooks, experiments, and documentation for the 6.5930 course project using AccelForge.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher (Python 3.12 recommended)
- Git

### Quick Setup (Recommended)

1. Clone both repositories:
```bash
# Clone the project repo
git clone https://github.com/maya486/6.5930-project.git

# Clone the AccelForge fork
git clone https://github.com/maya486/accelforge.git
```

Your directory structure should look like:
```
parent-directory/
├── 6.5930-project/    (this repo)
└── accelforge/        (forked AccelForge)
```

2. Run the setup script:
```bash
cd 6.5930-project
./setup.sh
```

This will automatically:
- Create a virtual environment at `../accelforge_env`
- Install all dependencies
- Install AccelForge in editable mode
- Install Jupyter

3. Start Jupyter:
```bash
./start_jupyter.sh
```

### Manual Setup

If you prefer to set up manually or the script doesn't work:

1. Clone both repositories (as above)

2. Create and activate a virtual environment:
```bash
cd 6.5930-project
python3 -m venv ../accelforge_env
source ../accelforge_env/bin/activate  # On Unix/macOS
# OR
..\accelforge_env\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install --only-binary :all: numba  # Install numba from binary first
pip install -e ../accelforge
pip install -r requirements.txt
```

4. Start Jupyter:
```bash
source ../accelforge_env/bin/activate  # Activate environment
jupyter notebook
```

## Project Structure

```
6.5930-project/
├── notebooks/       # Jupyter notebooks for experiments
├── data/           # Data files
├── docs/           # Documentation and reports
├── experiments/    # Experimental code and scripts
└── README.md       # This file
```

## Working with AccelForge

Since AccelForge is installed in editable mode, any changes you make to the AccelForge source code will be immediately reflected in your notebooks after restarting the kernel.

### Modifying AccelForge
1. Edit files in `../accelforge/accelforge/`
2. Restart your Jupyter kernel
3. Changes will be reflected immediately

### Syncing with Upstream AccelForge
To get updates from the original AccelForge repository:
```bash
cd ../accelforge
git fetch upstream
git merge upstream/main
git push origin main
```

## Collaboration

When working with collaborators:
1. They should clone both this repo and your AccelForge fork
2. Follow the installation instructions above
3. Push/pull changes to both repositories as needed

## License

This project uses AccelForge, which has its own license. See the AccelForge repository for details.
