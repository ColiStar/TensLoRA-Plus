import time
import subprocess
import os
import sys
import threading
import queue
import re

# Configuration
# ----------------------------------------------------------------
# Shared Job Queue
# Phase 7: Base LR 1e-5
JOB_QUEUE = queue.Queue()
COMPLETED_LOG_FILE = "scheduler_v3_completed_runs.txt"

jobs = [
    # --- Baseline (Must run first, has metrics) ---
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --compute-detailed-metrics \
        --use-amp \
        --use-wandb \
        --run-name "phase7_rank8_baseline_lr1e-5" \
        --seed 42""",

    # --- LR Experiments (Core) ---
    # Core x4
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --lr-core-mult 4.0 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_core_x4_lr1e-5" \
        --seed 42""",
    # Core x8
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --lr-core-mult 8.0 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_core_x8_lr1e-5" \
        --seed 42""",
    # Core x16
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --lr-core-mult 16.0 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_core_x16_lr1e-5" \
        --seed 42""",

    # --- Rank Experiments ---
    # Input Rank 2
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 2_8_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_input_2_lr1e-5" \
        --seed 42""",
    # Input Rank 4
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 4_8_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_input_4_lr1e-5" \
        --seed 42""",
    # QKV Rank 2
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_8_2_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_qkv_2_lr1e-5" \
        --seed 42""",
    # QKV Rank 4
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_8_4_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_qkv_4_lr1e-5" \
        --seed 42""",
    # Head Rank 16
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_8_16_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_head_16_lr1e-5" \
        --seed 42""",
    # HeadDim Rank 16
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac tucker \
        --n-components 8_16_8_8_8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_headdim_16_lr1e-5" \
        --seed 42""",

    # --- Variants ---
    # LoRA HF
    """python train_scripts/train_roberta.py \
        lora_hf \
        --n-components 8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_lora_hf_lr1e-5" \
        --seed 42""",
    # CP LoRA
    """python train_scripts/train_roberta.py \
        tenslora \
        --tensor-method att_qkv_depth \
        --tensor-fac cp \
        --n-components 8 \
        --lr 1e-5 \
        --n-epochs 10 \
        --batch-size 16 \
        --scaling 4 \
        --use-amp \
        --use-wandb \
        --run-name "phase7_cp_lora_lr1e-5" \
        --seed 42"""
]

# ----------------------------------------------------------------

def get_completed_runs():
    """Read completed runs from log file."""
    if not os.path.exists(COMPLETED_LOG_FILE):
        return set()
    with open(COMPLETED_LOG_FILE, "r") as f:
        return set(line.strip() for line in f)

def log_completed_run(run_name):
    """Append completed run to log file."""
    with open(COMPLETED_LOG_FILE, "a") as f:
        f.write(run_name + "\n")

def extract_run_name(cmd):
    """Extract run_name from command string."""
    match = re.search(r'--run-name\s+"([^"]+)"', cmd)
    if match:
        return match.group(1)
    return None

# Load completed runs
completed_runs = get_completed_runs()
print(f"Found {len(completed_runs)} completed runs: {completed_runs}")

# Populate Queue, skipping completed ones
for job in jobs:
    run_name = extract_run_name(job)
    if run_name and run_name in completed_runs:
        print(f"Skipping completed job: {run_name}")
        continue
    JOB_QUEUE.put(job)

# ----------------------------------------------------------------

def get_gpu_memory_usage(gpu_id):
    """Get used memory in MiB for a specific GPU using nvidia-smi."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", str(gpu_id)],
            encoding="utf-8"
        )
        return int(result.strip())
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return 99999 # Assume full if error

def worker(gpu_id):
    print(f"[GPU {gpu_id}] Worker started.")
    
    while not JOB_QUEUE.empty():
        # 1. Check GPU Memory
        mem_used = get_gpu_memory_usage(gpu_id)
        if mem_used > 2000: # Wait if > 2GB used
            print(f"[GPU {gpu_id}] Busy ({mem_used}MiB). Waiting...")
            time.sleep(60)
            continue
            
        # 2. Get Job
        try:
            # Non-blocking get, but we checked empty() above
            # Race condition possible but Queue is thread-safe
            cmd = JOB_QUEUE.get(block=False)
        except queue.Empty:
            break
            
        # 3. Run Job
        run_name = extract_run_name(cmd)
        print(f"[GPU {gpu_id}] Starting job: {run_name}")
        
        # Correctly format command: export first, then CUDA_VISIBLE_DEVICES python ...
        full_cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        
        try:
            subprocess.run(full_cmd, shell=True, check=True, executable="/bin/bash")
            print(f"[GPU {gpu_id}] Job finished successfully: {run_name}")
            if run_name:
                log_completed_run(run_name)
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Job failed: {e}")
            
        JOB_QUEUE.task_done()
        
        # Small pause between jobs
        time.sleep(10)

    print(f"[GPU {gpu_id}] No more jobs in queue. Exiting.")

def main():
    # Create threads for GPU 0 and GPU 1
    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    print("All tasks completed.")

if __name__ == "__main__":
    main()
