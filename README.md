# 6.5930 Course Project

This repository contains notebooks, experiments, and documentation for the 6.5930 course project using AccelForge.

## Setup Instructions

### Prerequisites
- Python 3.12 or higher
- Git

### Installation

1. Clone this repository:
```bash
git clone https://github.com/maya486/6.5930-project.git
cd 6.5930-project
```

2. Clone the AccelForge fork:
```bash
cd ..
git clone https://github.com/maya486/accelforge.git
```

3. Create and activate a virtual environment:
```bash
python3.12 -m venv accelforge_env
source accelforge_env/bin/activate  # On Unix/macOS
# OR
accelforge_env\Scripts\activate  # On Windows
```

4. Install AccelForge in editable mode:
```bash
pip install --only-binary :all: numba  # Install numba from binary first
pip install -e ../accelforge
```

5. Install Jupyter:
```bash
pip install jupyter ipykernel
```

### Running Jupyter

From the project directory:
```bash
source ../accelforge_env/bin/activate  # Activate environment
jupyter notebook
# OR
jupyter lab
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
