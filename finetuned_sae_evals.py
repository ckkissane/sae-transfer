# %%
from utils import *
from sae_lens.sae import SAE
import wandb
import argparse
import torch
from functools import partial
from typing import Any, Mapping
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from sae_lens.training.activations_store import ActivationsStore

# %%
# SAE Recons loss eval helper functions (cribbing heavily from SAELens)
@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    model_kwargs: Mapping[str, Any] = {},
    ignore_outliers: bool = False,
    outlier_threshold: int = 100_000,
):
    hook_name = sae.cfg.hook_name
    loss = model(batch_tokens, return_type="loss", **model_kwargs)

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)
    
        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.        
        if ignore_outliers:
            outlier_norm_mask = activations.norm(dim=-1) > outlier_threshold
            activations[~outlier_norm_mask, :] = sae.decode(sae.encode(activations)).to(activations.dtype)[~outlier_norm_mask, :]
        else:
            activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        
        return activations.to(original_device)

    def zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)


    replacement_hook = standard_replacement_hook
    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        **model_kwargs,
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        **model_kwargs,
    )

    div_val = zero_abl_loss - loss
    div_val[torch.abs(div_val) < 0.0001] = 1.0

    score = (zero_abl_loss - recons_loss) / div_val

    return score, loss, recons_loss, zero_abl_loss

def get_recons_loss_batches(
    batches_cpu: list[torch.Tensor],
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    ignore_outliers: bool = False,
    outlier_threshold: int = 100_000,
):
    ce_rec_store = torch.zeros(len(batches_cpu), device=sae.cfg.device)
    clean_loss_store = torch.zeros(len(batches_cpu), device=sae.cfg.device)
    sae_recons_loss_store = torch.zeros(len(batches_cpu), device=sae.cfg.device)
    zero_abl_loss_store = torch.zeros(len(batches_cpu), device=sae.cfg.device)
    for i in tqdm(range(len(batches_cpu))):
        all_tokens = batches_cpu[i]
        ce_rec, clean_loss, sae_recons_loss, zero_abl_loss = get_recons_loss(
            sae,
            model,
            all_tokens,
            activation_store,
            ignore_outliers=ignore_outliers,
            outlier_threshold=outlier_threshold,
        )
        ce_rec_store[i] = ce_rec.mean().item()
        clean_loss_store[i] = clean_loss.mean().item()
        sae_recons_loss_store[i] = sae_recons_loss.mean().item()
        zero_abl_loss_store[i] = zero_abl_loss.mean().item()
    return ce_rec_store.mean().item(), clean_loss_store.mean().item(), sae_recons_loss_store.mean().item(), zero_abl_loss_store.mean().item()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--wandb_entity',
        help='Wandb entity', default="sae-finetuning")
    parser.add_argument(
        '--eval_batch_size_prompts',
        help='Batch size when running evals', type=int)
    parser.add_argument(
        '--eval_batches',
        help='Number of batches when running evals', type=int)
    parser.add_argument(
        '--outlier_threshold',
        help='Activations with scaled norm above this are considered outliers', type=int, default=100_000)
    parser.add_argument(
        '--num_act_batches',
        help='Number of batches of activations running stuff like MSE and L0', type=int)
    parser.add_argument(
        '--ignore_outliers', action='store_true',
        help='whether to ignore outlier norms when calculating recons loss')
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    if args.ignore_outliers:
        print(f"Ignoring outliers. Outlier threshold: {args.outlier_threshold}")

    # Get model
    assert args.model_name in ["qwen1.5-0.5B-chat", "mistral-7B-instruct", "gemma-2b-it"], "Only supporting chat models for now"
    model = HookedTransformer.from_pretrained(args.model_name)
    print(f"Loaded model {args.model_name}")

    entity = args.wandb_entity
    
    if args.model_name == "qwen1.5-0.5B-chat":
        project = "qwen-500M-base-finetuning"
        artifact = "sae_qwen1.5-0.5b_blocks.13.hook_resid_pre_32768:v6"
    elif args.model_name == "mistral-7B-instruct":
        project = "mistral-7B-base-finetuning"
        artifact = "sae_mistral-7b_blocks.16.hook_resid_pre_131072:v1"
    elif args.model_name == "gemma-2b-it":
        raise NotImplementedError("gemma-2b-it not supported yet")
    else:
        raise ValueError(f"Invalid model: {args.model_name}. Probably just add it to the code")
    
    path = f"./artifacts/{artifact}"
    
    artifact_path = f"{entity}/{project}/{artifact}"
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact.download()
    
    sae = SAE.load_from_pretrained(path, device=device)
    print(f"Loaded SAE from {path}")

    # get activations store
    act_store = ActivationsStore.from_sae(model, sae)
    
    if args.model_name == "qwen1.5-0.5B-chat":
        norm_scaling_factor = 1.573157268076237
    elif args.model_name == "mistral-7B-instruct":
        norm_scaling_factor = 7.931667509979551
    elif args.model_name == "gemma-2b-it":
        norm_scaling_factor = 2.419691849168719
    else:
        raise ValueError(f"Invalid model: {args.model_name}. Probably just add it to the code")
    
    act_store.estimated_norm_scaling_factor = norm_scaling_factor
    
    # Run CE evals
    eval_batch_size_prompts = args.eval_batch_size_prompts

    batches_cpu = []
    num_batches = args.eval_batches
    for i in range(num_batches):
        batches_cpu.append(act_store.get_batch_tokens(eval_batch_size_prompts).cpu())

    print(f"Running evals on {sum(tokens.numel() for tokens in batches_cpu)} tokens")

    ce_rec, clean_loss, sae_recons_loss, zero_abl_loss = get_recons_loss_batches(
        batches_cpu,
        sae,
        model,
        act_store,
        ignore_outliers=args.ignore_outliers,
        outlier_threshold=args.outlier_threshold,
    )
    print(f"CE_REC: {ce_rec}")
    print(f"Clean loss: {clean_loss}")
    print(f"SAE recons loss: {sae_recons_loss}")
    print(f"Zero abl loss: {zero_abl_loss}")
    
    # MSE / L0 / etc Evals


    act_store = ActivationsStore.from_sae(model, sae) 
    
    act_store.estimated_norm_scaling_factor = norm_scaling_factor
    
    # Get MSE
    print(f"Running MSE / L0 / Explained Variance on {args.num_act_batches} batches")
    mse = get_mse_multiple_batches(
        sae, 
        act_store, 
        args.num_act_batches,
        ignore_outliers=args.ignore_outliers,
        outlier_threshold=args.outlier_threshold,
    )
    print(f"MSE: {mse}")
   
    # Get LO
    avg_num_firing = get_l0_multiple_batches(
        sae, 
        act_store, 
        args.num_act_batches,
        ignore_outliers=args.ignore_outliers,
        outlier_threshold=args.outlier_threshold,
    )
    print("L0", avg_num_firing)
    
    # Explained Variance
    explained_variance = get_explained_variance_multiple_batches(
        sae, 
        act_store, 
        args.num_act_batches,
        ignore_outliers=args.ignore_outliers,
        outlier_threshold=args.outlier_threshold,
    )
    print("Explained variance", explained_variance)
    
    records = {
        "model_name": args.model_name,
        "L0": avg_num_firing,
        "ce_rec": ce_rec,
        "clean_loss": clean_loss,
        "sae_recons_loss": sae_recons_loss,
        "zero_abl_loss": zero_abl_loss,
        "Explained Variance": explained_variance,
        "MSE": mse,
        "ignore_outliers": args.ignore_outliers,
        "eval_tokens": sum(tokens.numel() for tokens in batches_cpu),
        "MSE/L0/Explained Variance batches": args.num_act_batches,
        "Outlier threshold": args.outlier_threshold,
        "artifact_path": artifact_path
    }

    results_file = "./results/sae_finetuning_evals.json"

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Append the new results
    all_results.append(records)

    # Save all results back to the file
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_file}")
