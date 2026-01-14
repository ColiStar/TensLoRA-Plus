# TensLoRA+: Heterogeneous Allocation of Rank and Learning Rate
![Python](https://img.shields.io/badge/Language-Python_3.8%2B-blue?logo=python&logoColor=white)
![Topic](https://img.shields.io/badge/Topic-PEFT-orange)
[![Model: RoBERTa](https://img.shields.io/badge/Model-RoBERTa-blue)](https://huggingface.co/roberta-base)
[![Task: CoLA](https://img.shields.io/badge/Task-CoLA-green)](https://gluebenchmark.com/tasks)

**TensLoRA+** is an extension of the **TensLoRA** framework, introducing **Heterogeneous Allocation** strategies for both Rank and Learning Rate based on tensor spectral analysis.

This project bridges the gap between **Tensor Decomposition** methods and the optimization insights from **LoRA+**, proposing a theoretically grounded approach to allocate parameter budget and optimization resources where they matter most.

---

## 📖 Background & Motivation

To understand the contribution of TensLoRA+, we must first revisit the evolution of Parameter-Efficient Fine-Tuning (PEFT).

### 1. The Foundation: LoRA
Low-Rank Adaptation (LoRA) freezes the pre-trained weights $W_0$ and injects trainable rank decomposition matrices:
$$W = W_0 + \Delta W = W_0 + B A$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. This significantly reduces the number of trainable parameters compared to full fine-tuning.

### 2. The Evolution: TensLoRA (Tensor Decomposition)
While LoRA is efficient, it treats weights as 2D matrices. **TensLoRA** (Marmoret et al., 2025) argues that neural networks have multi-dimensional correlations (e.g., across layers, heads, and depth).
By reshaping $\Delta W$ into a tensor and applying **Tucker** or **CP Decomposition**, TensLoRA shares information across these dimensions, achieving even higher compression rates than standard LoRA.

### 3. What's the plus in TensLoRA+?
Despite its efficiency, standard TensLoRA treats all tensor components equally. We identified two critical limitations:

#### A. The Optimization Imbalance (Inspired by LoRA+)
Recent work in **LoRA+** (Hayou et al., 2024) proved that the two matrices in LoRA ($A$ and $B$) serve different roles:
* Matrix $B$ (initialized to zero) determines the direction of the update and requires a **higher learning rate** for efficient feature learning.
* Matrix $A$ (random initialization) requires a standard learning rate.

**Our Hypothesis**: In the context of Tucker Decomposition, the **Core Tensor ($\mathcal{C}$)** is mathematically analogous to LoRA's matrix $B$. It is initialized to zero and acts as the "hub" aggregating interactions between factors.
> **Therefore, the Core Tensor requires a significantly larger learning rate than the Factor matrices to maximize performance.**

#### B. The Rank Allocation Gap
While the TensLoRA framework theoretically supports flexible rank configurations, **the original work did not experimentally explore how to systematically vary rank across different tensor modes.**
> **Therefore, we did analysis and propose Heterogeneous Rank Allocation: a strategy to dynamically assign rank based on SVD Entropy and Energy metrics.**

---

## ⚙️ Methodology

### 1. Heterogeneous Learning Rate ($\lambda$)
We introduce a scalar multiplier $\lambda > 1$ specifically for the Core Tensor optimization:
$$\eta_{\text{core}} = \lambda \cdot \eta_{\text{factors}}$$
This allows the "hub" of the tensor to update rapidly while keeping the "spokes" (factors) stable.

### 2. Metric-Driven Rank Allocation
We determine the optimal rank for each mode by analyzing the pre-trained weight updates using:
* **SVD Entropy**: Measures information spread. High entropy $\rightarrow$ Needs higher rank.
* **Top-1 Energy**: Measures redundancy. High energy concentration $\rightarrow$ Can be compressed.

## Project Structure

```
TensLoRA/
├── tenslora/                  # Core Library (Tensor Layers & Operations)
├── tools/                     # [NEW] Utility Tools
│   └── experiment_scheduler.py # Automated Multi-GPU Experiment Scheduler
├── docs/                      # [NEW] Documentation & Analysis
│   ├── reports/               
├── train_scripts/             # Training Scripts
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

To run the **TensLoRA+ Auto-Scheduler** (manages queue across multiple GPUs)(modify the script to fit your needs):

```bash
python tools/experiment_scheduler.py
```
*   This tool automatically balances jobs across available GPUs, monitoring memory usage and handling job queues for large-scale ablation studies.

## 📊 Experimental Results

**Setup**:
* **Model**: RoBERTa-base
* **Dataset**: GLUE / CoLA
* **Baseline**: Standard Tucker-LoRA (Uniform Rank 8, Uniform LR)

### Experiment 1: Heterogeneous Learning Rate (Performance Validation)
We tested varying $\lambda$ values to validate the Core Tensor hypothesis.

| Experiment | Learning Rate Schedule | Accuracy | MCC Score | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | $\lambda = 1$ (Uniform) | 80.63% | 0.5255 | - |
| **TensLoRA+** | $\lambda = 4$ | 80.92% | 0.5536 | +5.3% |
| **TensLoRA+** | $\lambda = 8$ | 81.59% | 0.5656 | +7.6% |
| **TensLoRA+** | **$\lambda = 16$** | **81.87%** | **0.5658** | **+7.7%** |

> The results support our hypothesis. Increasing the learning rate ratio for the Core tensor yields consistently faster convergence and higher final performance, mirroring the findings of LoRA+.

### Experiment 2: Rank Sensitivity Analysis
We conducted an ablation study on rank allocation across different tensor modes (Input, QKV, Heads).

* **Critical Modes (Input/QKV)**: These modes exhibited high SVD Entropy. Compressing them (Rank $4 \to 2$) caused a **performance collapse** (MCC drop > 50%).
* **Redundant Modes**: Increasing rank in the 'Head' dimension did not improve MCC, suggesting that the model overfits when given excess capacity in dimensions that lack intrinsic information density.

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
