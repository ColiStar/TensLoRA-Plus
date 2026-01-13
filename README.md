# TensLoRA+: Heterogeneous Allocation of Rank and Learning Rate

TensLoRA+ is an extension of the **TensLoRA** framework (originally proposed by [Axel Marmoret et al.](#citation)), introducing **Heterogeneous Allocation of Rank and Learning Rate** based on tensor decomposition metrics.

While the original TensLoRA introduced tensor-based alternatives to Low-Rank Adaptation (LoRA), **TensLoRA+** specifically addresses the observation that not all tensor modes contribute equally to model performance. By analyzing metrics such as **Gradient Norm** and **SVD Entropy**, we implement a strategy to allocate more parameters (Rank) and optimization budget (Learning Rate) to the most critical components.

-----

## Key Features

### 1. TensLoRA Foundation
*   **Unified Framework**: Systematically explores different ways to tensorize LoRA updates (Tucker, CP).
*   **Parameter Efficiency**: Captures high-order correlations between attention heads, layers, and projections.

### 2. TensLoRA+ Contributions (New)
*   **Heterogeneous Learning Rates**:
    *   Applies distinct learning rates to the **Core Tensor** (which acts as a global information hub) versus **Factor Matrices**.
    *   **Inspiration**: This approach adapts the findings from **LoRA+** (Hayou et al., 2024), which showed that using a higher learning rate for the "B" matrix (analogous to our Core) significantly improves training efficiency.
    *   **Insight**: The Core tensor requires a larger learning rate multiplier to effectively aggregate features from different modes.
*   **Heterogeneous Rank Allocation**:
    *   Allocates Rank dynamically based on **SVD Entropy** (Information Density) and **Top-1 Energy** (Redundancy).
    *   **Strategy**: Modes with high entropy (complex information) receive higher rank, while redundant modes (e.g., certain layers) are compressed to rank as low as 2 without performance loss.
*   **Metric-Driven Optimization**:
    *   Provides tools to monitor **Gradient Norm** and **Eigenvalue Spectra** during training to guide hyperparameter tuning.

-----

## Project Structure

```
TensLoRA/
├── tenslora/                  # Core Library (Tensor Layers & Operations)
├── tools/                     # [NEW] Utility Tools
│   └── experiment_scheduler.py # Automated Multi-GPU Experiment Scheduler
├── docs/                      # [NEW] Documentation & Analysis
│   ├── reports/               # Detailed Performance Reports
│   └── project_motivation.md  # Deep dive into Heterogeneous Rank/LR Logic
├── train_scripts/             # Training Scripts (RoBERTa, ViT)
└── ...
```

-----

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/TensLoRA-Plus.git
cd TensLoRA-Plus
pip install -e .
```

### Running Experiments

To run a standard TensLoRA experiment:

```bash
python train_scripts/train_roberta.py tenslora --tensor-method att_qkv_depth --n-components 8_8_8_8_8
```

To run the **TensLoRA+ Auto-Scheduler** (manages queue across multiple GPUs):

```bash
python tools/experiment_scheduler.py
```
*   This tool automatically balances jobs across available GPUs, monitoring memory usage and handling job queues for large-scale ablation studies.

-----

## Experimental Results

Our detailed report can be found in [`docs/reports/ECE 273 Final Report.pdf`](docs/reports/ECE%20273%20Final%20Report%20(1).pdf).

**Key Findings:**
1.  **Core Importance**: The Core tensor in Tucker decomposition is highly sensitive to learning rate. Increasing its LR multiplier significantly accelerates convergence.
2.  **Redundancy in Layers**: We found that the `Layers` mode often exhibits low Effective Rank, allowing for aggressive compression (rank 4 -> 2) without accuracy loss.

-----

## Citation

This project builds upon the original TensLoRA work. If you use this codebase, please cite the original paper:

```bibtex
@article{marmoret2025tenslora,
  title={{TensLoRA}: Tensor Alternatives for Low-Rank Adaptation},
  author={Marmoret, Axel and Bensaid, Reda and Lys, Jonathan and Gripon, Vincent and Leduc-Primeau, Fran\c{c}ois},
  journal={arXiv preprint arXiv:TODO},
  year={2025}
}

@article{hayou2024loraplus,
  title={LoRA+: Efficient Low Rank Adaptation of Large Models},
  author={Hayou, Soufiane and Ghosh, Nikhil and Yu, Bin},
  journal={arXiv preprint arXiv:2402.12354},
  year={2024}
}
```

For the **Heterogeneous Allocation** extensions (TensLoRA+), please refer to the documentation in this repository.