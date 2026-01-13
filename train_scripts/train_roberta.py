"""
Standalone script to train with tensorlora on ROBERTa models.
Does not support training on multiple GPUs.
"""

import copy
import os
import random
import typer
from tqdm import tqdm
import wandb

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

# Reduce CUDA fragmentation per https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_scheduler

from tenslora.adapters.add_lora_to_model import lora_router
from tenslora.datasets_handler import get_glue_dataset
from tenslora.utils.parameter_count import count_trainable_parameters, predict_tenslora_parameters

CACHE_DIR = ".cache"
HELP_PANEL_NAME_1 = "Training Parameters"
HELP_PANEL_NAME_2 = "LORA Parameters"

# Default to the Hugging Face repo id so the weights can be downloaded automatically.
DEFAULT_MODEL_PATH = "FacebookAI/roberta-base"

# Wrapper logic

def get_classifier(
        num_classes: int, 
        lora_type: str, 
        n_components, 
        tensor_method: str = None, 
        tensor_fac: str = "tucker",
        tensor_init: str = "orthogonal",
        dropout_prob=0.0,
        init_from_saved_tensors=False,
        tensor_path=None,
        tensor_persisted_name=None,
        scaling=1,
        seed=0,
        model_path=DEFAULT_MODEL_PATH,
    ):
    model: RobertaForSequenceClassification = RobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes,
        problem_type="single_label_classification",
        cache_dir=CACHE_DIR,
    )

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")

    if lora_type == "tenslora" and tensor_method is None:
        raise ValueError("Tensor method must be specified when using tenslora.")

    kwargs_ = (
        {
            "tensor_fac": tensor_fac,
            "tensor_init": tensor_init,
            "tensor_method": tensor_method,
        }
        if lora_type == "tenslora"
        else {}
    )

    lora_model = lora_router(
        model=model,
        lora_type=lora_type,
        n_components=n_components,
        model_type="roberta",
        input_dim=768,
        output_dim=768,
        scaling=scaling,
        ## If you need to change these parameters, uncomment them:
        # dropout_prob=dropout_prob,
        # init_from_saved_tensors=init_from_saved_tensors,
        # tensor_path=tensor_path,
        # tensor_persisted_name=tensor_persisted_name,
        seed=seed,
        **kwargs_,
    )

    return lora_model


def get_factor_names(tensor_method):
    factor_names_map = {
        "att_qkv_depth": ["Input", "HeadDim", "Heads", "QKV", "Layers"],
        "att_qkv": ["Input", "HeadDim", "Heads", "QKV"],
        "att_depth": ["Input", "HeadDim", "Heads", "Layers"],
        "qkv_depth": ["Input", "Output", "QKV", "Layers"],
        "depth": ["Input", "Output", "Layers"],
        "qkv": ["Input", "Output", "QKV"],
        "att": ["Input", "HeadDim", "Heads"],
    }
    return factor_names_map.get(tensor_method, [])

def compute_grad_norms(model, factor_names):
    metrics = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
            
        if "tucker_core" in name:
            metrics[f"grad_norm/core"] = param.grad.norm().item()
        elif "tucker_factors" in name:
            try:
                # name format example: ...tucker_factors.0...
                parts = name.split(".")
                idx = parts.index("tucker_factors") + 1
                factor_idx = int(parts[idx])
                
                if factor_idx < len(factor_names):
                    fname = factor_names[factor_idx]
                else:
                    fname = f"Factor_{factor_idx}"
                
                metrics[f"grad_norm/{fname}"] = param.grad.norm().item()
            except (ValueError, IndexError):
                pass
    return metrics

def compute_svd_metrics(model, factor_names):
    metrics = {}
    # 2. SVD Spectrum of Core
    core_param = None
    for name, param in model.named_parameters():
        if "tucker_core" in name:
            core_param = param
            break
            
    if core_param is not None:
        core_tensor = core_param.data
        n_modes = core_tensor.ndim
        
        for i in range(n_modes):
            try:
                # Unfold: Move mode i to front, flatten rest
                # shape: (dim_i, product_of_others)
                unfolded = core_tensor.permute(i, *[j for j in range(n_modes) if j != i]).reshape(core_tensor.shape[i], -1)
                
                # SVD
                s = torch.linalg.svdvals(unfolded)
                
                # Normalize energy
                energy = s**2
                total_energy = energy.sum() + 1e-9
                normalized_energy = energy / total_energy
                
                fname = factor_names[i] if i < len(factor_names) else str(i)
                
                # Log top-1 energy ratio
                metrics[f"svd_energy_top1/{fname}"] = normalized_energy[0].item()
                
                # Log entropy
                entropy = -torch.sum(normalized_energy * torch.log(normalized_energy + 1e-9)).item()
                metrics[f"svd_entropy/{fname}"] = entropy
                
            except Exception:
                pass

    return metrics


def main(  # noqa: C901, PLR0912, PLR0915
    # LoRA parameters
    lora_type: str = typer.Argument(..., help="Type of LoRA to use"),
    scaling: float = typer.Option(4.0, help="LoRA alpha value for scaling the LoRA weights"),
    # TensLoRA parameters
    tensor_method: str = typer.Option(None, help="Method for tensor decomposition (e.g., 'att', 'qkv', 'depth')"),
    tensor_fac: str = typer.Option(
        "tucker",
        help="Tensor factorization method for TensLoRA ('tucker' or 'cp').",
    ),
    tensor_init: str = typer.Option(
        "orthogonal",
        help="Tensor initialization for TensLoRA (orthogonal, normal, kaiming_uniform).",
    ),
    n_components: str = typer.Option("4",help="Number of components for TensLoRA. Expected to be a string, to pass either int or list of int. Use underscores to separate multiple components (e.g., '4_8_16')"),
    # Dataset
    dataset: str = typer.Option("cola", help="Dataset to use for training. Options: 'tldr', 'xsum'"),
    # Training parameters
    lr: float = typer.Option(5e-4, help="Learning rate for the optimizer"),
    n_epochs: int = typer.Option(10, help="Number of epochs to train"),
    batch_size: int = typer.Option(64, help="Batch size per GPU"),
    # weight_decay: float = typer.Option(0.01, help="Weight decay"),
    # dropout_prob: float = typer.Option(0.0, help="Dropout probability for LoRA layers"),
    seed: int = typer.Option(None, help="Random seed for reproducibility"),
    # Other parameters
    test: bool = typer.Option(False, help="Run in test mode"),
    run_name: str = typer.Option("tensorlora-llm", help="Run name for logging"),
    use_wandb: bool = typer.Option(False, help="Enable Wandb"),
    use_amp: bool = typer.Option(False, help="Enable torch.cuda.amp mixed precision to save memory"),
    # Differential LR parameters
    lr_core_mult: float = typer.Option(1.0, help="LR multiplier for Tucker Core"),
    lr_input_mult: float = typer.Option(1.0, help="LR multiplier for Input Factor"),
    lr_headdim_mult: float = typer.Option(1.0, help="LR multiplier for HeadDim Factor"),
    lr_heads_mult: float = typer.Option(1.0, help="LR multiplier for Heads Factor"),
    lr_qkv_mult: float = typer.Option(1.0, help="LR multiplier for QKV Factor"),
    lr_layers_mult: float = typer.Option(1.0, help="LR multiplier for Layers Factor"),
    lr_output_mult: float = typer.Option(1.0, help="LR multiplier for Output Factor (if applicable)"),
    # Metrics
    compute_detailed_metrics: bool = typer.Option(False, help="Compute expensive metrics like Grad Norm and SVD Entropy"),
):
    # Init wandb if needed
    if use_wandb:
        wandb.login()
        
    # Seed
    if seed is None:  # Fix the seed anyway
        print("No seed provided, using default seed 0.")
        seed = 0
    elif type(seed) is not int:
        raise ValueError(f"Seed must be an integer, got {type(seed)} instead.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        print("CUDA not available, falling back to CPU. Training will be much slower.")
    # torch.use_deterministic_algorithms(True)

    # Set the number of components in LoRA/TensLoRA
    if type(n_components) is str:
        n_components_str = copy.deepcopy(n_components)
        # Parse n_components as a list of ints if underscores are present, else as a single int
        n_components = [int(x) for x in n_components.split("_")] if "_" in n_components else int(n_components)
    else:
        raise ValueError(f"Unsupported type for n_components: {type(n_components)}. Expected str.")

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )

    ## Dataset setup
    # Dataset parameters
    train_dataloader, validation_dataloader, num_classes = get_glue_dataset(tokenizer, batch_size, dataset, test=test)
    print("Dataset loaded successfully!")

    # Print experiment information
    print(
        f"{dataset} : Training params : lr: {lr} | n_epochs: {n_epochs}",
    )

    ## Model setup
    model = get_classifier(
        num_classes,
        lora_type, 
        n_components=n_components, 
        tensor_method=tensor_method, 
        tensor_fac=tensor_fac,
        tensor_init=tensor_init,
        seed=seed, 
        scaling=scaling,
    )
    print("Model loaded successfully!")

    # Print the LoRA type and parameters
    print_tensor = f" - {tensor_method}_{tensor_fac}_{tensor_init}" if tensor_method else ""
    print(f"{lora_type}{print_tensor} | n_components: {n_components_str} | seed: {seed} | scaling: {scaling}")

    # Quick test to see if the model works
    txt = "hello"
    inputs = tokenizer(txt, return_tensors="pt").input_ids
    outputs = model(inputs, output_hidden_states=True, return_dict=True)
    print("Model output:", outputs.hidden_states[-1])
    # Should return:
    # Model output: tensor([[[-0.0712,  0.0839,  0.0174,  ..., -0.0752, -0.0725, -0.0115],
    #     [-0.0242, -0.2140,  0.1199,  ..., -0.3125, -0.2366,  0.0626],
    #     [-0.0778,  0.0813, -0.0014,  ..., -0.1284, -0.0861, -0.0517]]],
    #   grad_fn=<NativeLayerNormBackward0>)

    ## Training setup
    # Training parameters
    n_steps = len(train_dataloader) * n_epochs
    warmup_ratio = 0.1
    num_decay_steps = min(500, int(n_steps * 0.15))

    # eval_ratio = 1/n_epochs
    log_ratio = 0.01

    log_every = max(1, int(n_steps * log_ratio))
    eval_every = max(1, len(train_dataloader))

    # Collect parameters for training
    parameters = []
    for name, param in model.named_parameters():
        if "lora" in name or "tenslora" in name or "classifier" in name:
            param.requires_grad = True  # Ensure LoRA parameters are trainable
            parameters.append(param)
            # print(f"Parameter {name} is trainable with shape {param.shape}")

        else:
            param.requires_grad = False

    # Count parameters
    adapters_trainable_params, trainable_params, all_params = count_trainable_parameters(model)
    non_trainable_params = all_params - trainable_params

    if lora_type == "tenslora" and tensor_fac == "tucker":  # Parameter count check only reliable for Tucker
        count_tenslora_params = predict_tenslora_parameters(
            method=tensor_method,
            tenslora_set_ranks=n_components,
            hidden_dim=768,
            layer=12,
            num_heads=12,
        )

        assert count_tenslora_params == adapters_trainable_params, (
            f"Counted TensLoRA parameters ({count_tenslora_params}) do not match adapter parameters ({adapters_trainable_params})."
        )

    print(
        f"Trainable params: {trainable_params} | All params: {all_params} | % of trainable: {100 * trainable_params / all_params:.3f}",
    )
    print(
        f"Adapter only params: {adapters_trainable_params} | % of trainable: {100 * adapters_trainable_params / all_params:.3f}",
    )

    ## Optimization setup
    ## Optimization setup
    # Optimizer
    
    current_factor_names = get_factor_names(tensor_method)
    
    # LR Multiplier Map
    lr_mult_map = {
        "Input": lr_input_mult,
        "HeadDim": lr_headdim_mult,
        "Heads": lr_heads_mult,
        "QKV": lr_qkv_mult,
        "Layers": lr_layers_mult,
        "Output": lr_output_mult,
    }

    optimizer_grouped_parameters = []
    
    # Helper to check if param is already added
    added_ids = set()

    # 1. Core
    core_params = []
    for name, param in model.named_parameters():
        if "tucker_core" in name and param.requires_grad:
            core_params.append(param)
            added_ids.add(id(param))
    if core_params:
        optimizer_grouped_parameters.append({
            "params": core_params,
            "lr": lr * lr_core_mult,
            "name": "tucker_core"
        })

    # 2. Factors
    for i, fname in enumerate(current_factor_names):
        factor_params = []
        mult = lr_mult_map.get(fname, 1.0)
        
        for name, param in model.named_parameters():
            if "tucker_factors" in name and param.requires_grad:
                # Check index
                try:
                    parts = name.split(".")
                    idx = parts.index("tucker_factors") + 1
                    factor_idx = int(parts[idx])
                    if factor_idx == i:
                        factor_params.append(param)
                        added_ids.add(id(param))
                except:
                    pass
        
        if factor_params:
            optimizer_grouped_parameters.append({
                "params": factor_params,
                "lr": lr * mult,
                "name": f"factor_{fname}"
            })

    # 3. Others (Classifier, etc.)
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in added_ids:
            other_params.append(param)
    
    if other_params:
        optimizer_grouped_parameters.append({
            "params": other_params,
            "lr": lr,
            "name": "others"
        })

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr, # Default LR, though groups override it
        fused=True,
    )

    # Learning rate scheduler
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(n_steps * warmup_ratio),
        num_training_steps=n_steps,
    )

    print("Optimizer and scheduler initialized.")

    device = torch.device("cuda" if cuda_available else "cpu")
    if cuda_available:
        print(f"DEBUG: Using Device: {device}")
        print(f"DEBUG: Current CUDA Device Index: {torch.cuda.current_device()}")
        print(f"DEBUG: Physical Device Name: {torch.cuda.get_device_name(0)}")

    amp_enabled = use_amp and device.type == "cuda"
    if use_amp and not amp_enabled:
        print("Warning: --use-amp requested but CUDA is unavailable. Disabling AMP.")

    scaler = GradScaler(enabled=amp_enabled)

    if use_wandb:
        wandb.init(
            project="tensorlora-llm",
            name=run_name,
            config={
                "learning_rate": lr,
                "n_components": n_components,
                "scaling": scaling,
                "seed": seed,
                "batch_size": batch_size,
                "lora_type": lora_type,
                "dataset": dataset,
                "tensor_method": tensor_method,
                "tensor_fac": tensor_fac,
                "tensor_init": tensor_init,
                "n_steps": n_steps,
                "warmup_ratio": warmup_ratio,
                "num_decay_steps": num_decay_steps,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "fraction_trainable": trainable_params / all_params,
                "adapter_parameters": adapters_trainable_params,
                "fraction_adapter": adapters_trainable_params / all_params,
                "model_name": DEFAULT_MODEL_PATH,
                "compute_detailed_metrics": compute_detailed_metrics,
            },
        )

        # Log adapter statistics once so they appear on the dashboard
        wandb.log(
            {
                "adapter_parameters": adapters_trainable_params,
                "fraction_adapter": adapters_trainable_params / all_params,
            },
            step=0,
        )

    # Training loop
    step = 0
    pbar = tqdm(total=n_steps, desc="Training")

    best_acc = 0
    best_mcc = 0

    model.to(device)
    
    # Initialize accumulator for gradient norms
    grad_norm_accumulator = {}

    while step < n_steps:
        model.train()
        for batch in train_dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):
                outputs = model(inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            # --- Enhanced Logging ---
            # 1. Compute Gradient Norms (Every Step)
            if step % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                
                log_dict = {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                }

                # 1. Compute Gradient Norms (Only every log_every steps)
                if compute_detailed_metrics and lora_type == "tenslora" and tensor_fac == "tucker":
                    grad_norms = compute_grad_norms(model, current_factor_names)
                    log_dict.update(grad_norms)
                    
                    # Accumulate for epoch stats (Optional: Sampling instead of every step)
                    for k, v in grad_norms.items():
                        if k not in grad_norm_accumulator:
                            grad_norm_accumulator[k] = []
                        grad_norm_accumulator[k].append(v)

                # 2. Log All Learning Rates
                for i, param_group in enumerate(optimizer.param_groups):
                    group_name = param_group.get("name", f"group_{i}")
                    log_dict[f"lr/{group_name}"] = param_group["lr"]

                # 3. Compute SVD Metrics (Only every log_every steps)
                if compute_detailed_metrics and lora_type == "tenslora" and tensor_fac == "tucker":
                     svd_metrics = compute_svd_metrics(model, current_factor_names)
                     log_dict.update(svd_metrics)


                if use_wandb:
                    wandb.log(
                        log_dict,
                        step=step,
                    )
                pbar.set_postfix({"loss": loss.item(), "step": step, "lr": lr})

            if step % eval_every == 0:
                model.eval()
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for val_batch in validation_dataloader:
                        val_inputs = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)

                        with autocast(enabled=amp_enabled):
                            val_outputs = model(val_inputs, attention_mask=val_attention_mask)
                        logits = val_outputs.logits
                        preds = torch.argmax(logits, dim=-1)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(val_labels.cpu().numpy())

                accuracy = accuracy_score(all_labels, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
                mcc = matthews_corrcoef(all_labels, all_preds)

                if accuracy > best_acc:
                    best_acc = accuracy
                if mcc > best_mcc:
                    best_mcc = mcc

                print(
                    f"Step {step}: Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}",
                    f" | Best Acc: {best_acc:.4f}, Best MCC: {best_mcc:.4f}",
                )

                if use_wandb:
                    wandb.log(
                        {
                            "eval/accuracy": accuracy,
                            "eval/precision": precision,
                            "eval/recall": recall,
                            "eval/f1": f1,
                            "eval/mcc": mcc,
                            "eval/best_accuracy": best_acc,
                            "eval/best_mcc": best_mcc,
                        },
                        step=step,
                    )

            pbar.update(1)
            step += 1
            if step >= n_steps:
                break

        # End of Epoch Logging (Mean/Max Grad Norms)
        if lora_type == "tenslora" and tensor_fac == "tucker" and use_wandb:
            epoch_stats = {}
            for k, v in grad_norm_accumulator.items():
                if v:
                    epoch_stats[f"{k}_mean"] = np.mean(v)
                    epoch_stats[f"{k}_max"] = np.max(v)
            wandb.log(epoch_stats, step=step)
            # Reset accumulator for next epoch
            grad_norm_accumulator = {}

    pbar.close()
    print("Training completed successfully!")

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings
    typer.run(main)
