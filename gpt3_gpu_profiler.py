"""
GPT-3 Workload GPU Profiler - Matching TPU v4i Specs

TPU v4i Architecture (from tpu_v4i.yaml):
- 4 spatial tiles (Z dimension)
- 128x128 Processing Elements per tile = 16,384 MACs per tile
- Total: 65,536 MACs across 4 tiles
- 128MB GlobalBuffer, 4MB LocalBuffer per tile
- MAC: 1.05 GHz, 0.084 pJ per operation

Workload (from gpt3_175B.yaml):
- B (batch): 1
- M (tokens): 8192
- H (heads): 96
- E (embedding per head): 128
- D (total embedding): 96*128 = 12,288

This script profiles the actual workload on GPU to get real scheduling data.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
import pandas as pd
import matplotlib.pyplot as plt
import json

class GPT3Workload(nn.Module):
    """
    Full GPT-3 workload matching gpt3_175B.yaml specifications.

    Uses ACTUAL dimensions from the YAML file.
    """

    def __init__(self, B=1, M=8192, H=96, E=128):
        super().__init__()

        # Exact dimensions from gpt3_175B.yaml
        self.B = B          # Batch size: 1
        self.M = M          # Tokens: 8192
        self.P = M          # Tokens (for attention): 8192
        self.H = H          # Heads: 96
        self.E = E          # Embedding per head: 128
        self.F = E          # 128
        self.D = H * E      # Total embedding: 12,288
        self.C = 4 * H * E  # FFN dimension: 49,152
        self.J = H * E      # Output: 12,288
        self.G = H * E      # Intermediate: 12,288

        print(f"Model Configuration (matching gpt3_175B.yaml):")
        print(f"  Batch (B): {self.B}")
        print(f"  Tokens (M, P): {self.M}")
        print(f"  Heads (H): {self.H}")
        print(f"  Embedding/head (E): {self.E}")
        print(f"  Total embedding (D = H*E): {self.D}")
        print(f"  FFN dimension (C = 4*H*E): {self.C}")
        print(f"\nTensor Sizes:")
        print(f"  Input I_in: [{B}, {M}, {self.D}]")
        print(f"  Q, K, V: [{B}, {M}, {H}, {E}]")
        print(f"  QK: [{B}, {M}, {self.P}, {H}]")
        print(f"  Weights WV, WK, WQ: [{H}, {E}, {self.D}]")

        # Initialize weights (matching YAML persistent_tensors)
        # Using fp16 to fit in GPU memory
        self.WV = nn.Parameter(torch.randn(H, E, self.D, dtype=torch.float16) * 0.01)
        self.WK = nn.Parameter(torch.randn(H, E, self.D, dtype=torch.float16) * 0.01)
        self.WQ = nn.Parameter(torch.randn(H, E, self.D, dtype=torch.float16) * 0.01)
        self.WZ = nn.Parameter(torch.randn(H, self.F, self.G, dtype=torch.float16) * 0.01)
        self.WFFA = nn.Parameter(torch.randn(self.G, self.C, dtype=torch.float16) * 0.01)
        self.WFFB = nn.Parameter(torch.randn(self.C, self.J, dtype=torch.float16) * 0.01)

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        total_size_mb = total_params * 2 / (1024**2)  # fp16 = 2 bytes
        print(f"\nModel Stats:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: {total_size_mb:.2f} MB (fp16)")

    def forward(self, I_in):
        """
        Forward pass matching gpt3_175B.yaml operations.

        Operations (from YAML):
        1. I[b, m, d] = I_in[b, m, d]                        # Copy
        2. V[b, m, h, e] = I[b, m, d] * WV[h, e, d]          # Value projection
        3. K[b, m, h, e] = I[b, m, d] * WK[h, e, d]          # Key projection
        4. Q[b, m, h, e] = I[b, m, d] * WQ[h, e, d]          # Query projection
        5. QK[b, m, p, h] = Q[b, m, h, e] * K[b, p, h, e]    # Attention scores
        6. QK_softmax[b, m, p, h] = QK[b, m, p, h]           # Softmax
        7. AV[b, m, h, f] = QK_softmax * V[b, p, h, f]       # Attention output
        8. Z[b, m, g] = AV[b, m, h, f] * WZ[h, f, g]         # Output projection
        9. FFA[b, m, c] = Z[b, m, g] * WFFA[g, c]            # Feed-forward A
        10. FFB[b, m, j] = FFA[b, m, c] * WFFB[c, j]         # Feed-forward B
        """

        # Op 1: I[b, m, d] = I_in[b, m, d] (copy operation)
        with record_function("01_I_copy"):
            I = I_in.clone()

        # Op 2: V[b, m, h, e] = I[b, m, d] * WV[h, e, d]
        with record_function("02_V_matmul"):
            V = torch.einsum('bmd,hed->bmhe', I, self.WV)

        # Op 3: K[b, m, h, e] = I[b, m, d] * WK[h, e, d]
        with record_function("03_K_matmul"):
            K = torch.einsum('bmd,hed->bmhe', I, self.WK)

        # Op 4: Q[b, m, h, e] = I[b, m, d] * WQ[h, e, d]
        with record_function("04_Q_matmul"):
            Q = torch.einsum('bmd,hed->bmhe', I, self.WQ)

        # Op 5: QK[b, m, p, h] = Q[b, m, h, e] * K[b, p, h, e]
        with record_function("05_QK_matmul"):
            QK = torch.einsum('bmhe,bphe->bmph', Q, K)

        # Op 6: QK_softmax[b, m, p, h] = softmax(QK)
        with record_function("06_QK_softmax"):
            QK_softmax = torch.softmax(QK, dim=2)

        # Op 7: AV[b, m, h, f] = QK_softmax * V
        with record_function("07_AV_matmul"):
            AV = torch.einsum('bmph,bphe->bmhe', QK_softmax, V)

        # Op 8: Z[b, m, g] = AV * WZ
        with record_function("08_Z_matmul"):
            Z = torch.einsum('bmhf,hfg->bmg', AV, self.WZ)

        # Op 9: FFA[b, m, c] = Z * WFFA
        with record_function("09_FFA_matmul"):
            FFA = torch.einsum('bmg,gc->bmc', Z, self.WFFA)

        # Op 10: FFB[b, m, j] = FFA * WFFB
        with record_function("10_FFB_matmul"):
            FFB = torch.einsum('bmc,cj->bmj', FFA, self.WFFB)

        return FFB


def profile_model_on_gpu(model, input_tensor, device='cuda', warmup_iters=3):
    """Profile model execution on GPU."""

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    print(f"\n{'='*70}")
    print(f"Profiling on {device.upper()}")
    print(f"{'='*70}")

    # Warmup
    print(f"\nWarming up ({warmup_iters} iterations)...")
    with torch.no_grad():
        for i in range(warmup_iters):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{warmup_iters} complete")

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Profile
    print("\nRunning profiler...")
    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            output = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

    print("✓ Profiling complete")
    return prof, output


def extract_schedule(prof, device='cuda'):
    """Extract operation schedule from profiler."""

    events = prof.key_averages()

    schedule = []
    for evt in events:
        # Only get our numbered operations
        if any(f"{i:02d}_" in evt.key for i in range(1, 11)):
            op_data = {
                'operation': evt.key,
                'cpu_time_ms': evt.cpu_time_total / 1000,
                'self_cpu_time_ms': evt.self_cpu_time_total / 1000,
                'cpu_memory_mb': evt.cpu_memory_usage / (1024**2),
                'calls': evt.count
            }

            if device == 'cuda' and hasattr(evt, 'cuda_time_total'):
                op_data['cuda_time_ms'] = evt.cuda_time_total / 1000
                op_data['self_cuda_time_ms'] = evt.self_cuda_time_total / 1000
                op_data['cuda_memory_mb'] = evt.cuda_memory_usage / (1024**2)

            schedule.append(op_data)

    df = pd.DataFrame(schedule)
    if not df.empty:
        df = df.sort_values('operation')

    return df


def visualize_schedule(schedule_df, device='cuda', output_file='gpt3_schedule.png'):
    """Create Gantt chart and performance visualization."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ops = [op.replace('_', ' ').title() for op in schedule_df['operation']]
    x = range(len(ops))

    # Plot 1: Execution Time
    ax1 = axes[0]
    if device == 'cuda' and 'cuda_time_ms' in schedule_df.columns:
        cuda_time = schedule_df['cuda_time_ms']
        ax1.bar(x, cuda_time, color='green', alpha=0.7, label='GPU Time')
    else:
        cpu_time = schedule_df['cpu_time_ms']
        ax1.bar(x, cpu_time, color='blue', alpha=0.7, label='CPU Time')

    ax1.set_xticks(x)
    ax1.set_xticklabels(ops, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('GPT-3 Operation Execution Time (Matching tpu_v4i.yaml)',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Memory Usage
    ax2 = axes[1]
    if device == 'cuda' and 'cuda_memory_mb' in schedule_df.columns:
        memory = schedule_df['cuda_memory_mb']
        ax2.bar(x, memory, color='orange', alpha=0.7)
        ylabel = 'GPU Memory (MB)'
    else:
        memory = schedule_df['cpu_memory_mb']
        ax2.bar(x, memory, color='purple', alpha=0.7)
        ylabel = 'CPU Memory (MB)'

    ax2.set_xticks(x)
    ax2.set_xticklabels(ops, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax2.set_title('Memory Usage per Operation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Cumulative Timeline
    ax3 = axes[2]
    if device == 'cuda' and 'cuda_time_ms' in schedule_df.columns:
        times = schedule_df['cuda_time_ms'].values
    else:
        times = schedule_df['cpu_time_ms'].values

    cumulative = [0]
    for t in times:
        cumulative.append(cumulative[-1] + t)

    for i, op in enumerate(ops):
        ax3.barh(i, times[i], left=cumulative[i], height=0.8,
                color=plt.cm.Set3(i % 12), edgecolor='black', linewidth=1)
        # Add operation label
        ax3.text(cumulative[i] + times[i]/2, i, op.split()[1],
                ha='center', va='center', fontsize=8, fontweight='bold')

    ax3.set_yticks(x)
    ax3.set_yticklabels(ops, fontsize=9)
    ax3.set_xlabel('Cumulative Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('Execution Timeline (Sequential Schedule)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")
    return fig


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("GPT-3 Workload Profiler - Matching TPU v4i Architecture")
    print("="*70)

    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        device = 'cpu'
        print("\n⚠ No GPU available, using CPU (will be slow)")

    # Create model with EXACT specs from gpt3_175B.yaml
    print(f"\n{'='*70}")
    print("Creating Model")
    print("="*70)
    model = GPT3Workload(B=1, M=8192, H=96, E=128)

    # Create input tensor
    print(f"\n{'='*70}")
    print("Creating Input")
    print("="*70)
    I_in = torch.randn(1, 8192, 12288, dtype=torch.float16)
    print(f"Input shape: {I_in.shape}")
    print(f"Input size: {I_in.numel() * 2 / 1e6:.2f} MB (fp16)")

    # Profile
    prof, output = profile_model_on_gpu(model, I_in, device=device)

    print(f"\n✓ Output shape: {output.shape}")

    # Extract schedule
    print(f"\n{'='*70}")
    print("Extracting Schedule")
    print("="*70)
    schedule_df = extract_schedule(prof, device=device)

    # Print schedule
    print("\n" + "="*70)
    print("EXECUTION SCHEDULE")
    print("="*70)
    print(schedule_df.to_string(index=False))

    # Summary statistics
    print(f"\n{'='*70}")
    print("Summary")
    print("="*70)
    if device == 'cuda' and 'cuda_time_ms' in schedule_df.columns:
        total_time = schedule_df['cuda_time_ms'].sum()
        print(f"Total GPU time: {total_time:.2f} ms")
    else:
        total_time = schedule_df['cpu_time_ms'].sum()
        print(f"Total CPU time: {total_time:.2f} ms")

    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

    # Save results
    schedule_df.to_csv('gpt3_gpu_schedule.csv', index=False)
    print("\n✓ Schedule saved to gpt3_gpu_schedule.csv")

    # Visualize
    visualize_schedule(schedule_df, device=device, output_file='gpt3_gpu_schedule.png')

    # Export trace
    prof.export_chrome_trace("gpt3_trace.json")
    print("✓ Chrome trace saved to gpt3_trace.json")
    print("  (View at chrome://tracing)")

    # Print detailed profiler output
    print(f"\n{'='*70}")
    print("Detailed Profiler Output (Top 30)")
    print("="*70)
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if device == 'cuda' else "cpu_time_total",
        row_limit=30
    ))
