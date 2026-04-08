#!/bin/bash
#SBATCH --job-name=gpt3_profiler
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=gpt3_profiler_%j.out
#SBATCH --error=gpt3_profiler_%j.err

# Load required modules for Python 3.10.8
module load deprecated-modules
module load gcc/12.2.0-x86_64
module load python/3.10.8-x86_64

# Load CUDA if available
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null

# Activate virtual environment
echo "Activating PyTorch environment..."
source ~/pytorch_env/bin/activate

# Verify activation
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Print GPU info
echo ""
echo "========================================"
echo "GPU Information"
echo "========================================"
nvidia-smi

# Print PyTorch info
echo ""
echo "========================================"
echo "PyTorch Information"
echo "========================================"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run the profiler
echo ""
echo "========================================"
echo "Running GPT-3 GPU Profiler"
echo "========================================"
python3 gpt3_gpu_profiler.py

# Check if profiler succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Profiler completed successfully"

    # Copy results to a timestamped directory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="results_${TIMESTAMP}"
    mkdir -p ${RESULTS_DIR}

    # Move output files if they exist
    [ -f gpt3_gpu_schedule.csv ] && mv gpt3_gpu_schedule.csv ${RESULTS_DIR}/
    [ -f gpt3_gpu_schedule.png ] && mv gpt3_gpu_schedule.png ${RESULTS_DIR}/
    [ -f gpt3_trace.json ] && mv gpt3_trace.json ${RESULTS_DIR}/

    echo ""
    echo "========================================"
    echo "Results saved to ${RESULTS_DIR}/"
    echo "========================================"
    ls -lh ${RESULTS_DIR}/
else
    echo ""
    echo "✗ Profiler failed with exit code $?"
    echo "Check the error output above"
fi
