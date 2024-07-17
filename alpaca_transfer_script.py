# %%
from utils import *
from datasets import load_dataset
import itertools
from transformer_lens import HookedTransformer
import wandb
from sae_lens.sae import SAE
from transformer_lens.hook_points import HookedRootModule
from typing import Any
import pickle

# %%
def batch_iterator_chat_completions(dataset_instructions, dataset_outputs, tokenize_instructions_fn, batch_size, eoi_toks, soi_toks, mask_type):
    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        loss_mask = inputs["attention_mask"].clone()
        # don't record loss on final token
        for b in range(loss_mask.shape[0]):
            last_idx = loss_mask[b].nonzero(as_tuple=True)[0].max()
            loss_mask[b, last_idx] = 0
            loss_mask[b, last_idx-1] = 0

        # also mask out all tokens before the eoi token region
        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):
                
                if inputs["input_ids"][b, i] == eoi_toks[0] and torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    if mask_type == "user_prompt":
                        loss_mask[b, :soi_toks.shape[0]] = 0
                        loss_mask[b, i:] = 0
                        break
                    elif mask_type == "control_tokens":
                        loss_mask = torch.zeros_like(loss_mask)
                        loss_mask[b, :soi_toks.shape[0]] = 1
                        loss_mask[b, i:i + eoi_toks.shape[0] - 1] = 1
                        break
                    elif mask_type == "rollout":
                        loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0 
                        break
                    else:
                        raise ValueError(f"Unknown mask type: {mask_type}")

        yield inputs, loss_mask 

# %%
def greedy_sample(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        logits = model(all_toks[:, :-max_tokens_generated + i])
        next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
        all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=False)

def generate_rollouts(chat_model, tokenize_instructions_fn, num_samples=10):
    """Yields batches from the Alpaca dataset."""

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)

    instructions, completions = [], []

    for i in tqdm(range(num_samples)):
        if dataset[i]['input'].strip() == '': # filter for instructions that do not have inputs
            instruction = dataset[i]['instruction']
            instructions.append(dataset[i]['instruction'])
            
            toks = tokenize_instructions_fn(instructions=[instruction])["input_ids"]
            generation = greedy_sample(
                chat_model,
                toks,
            )
            completions.extend(generation)

    return instructions, completions

def filter_completions(model_name, instructions, completions):
    if model_name == "qwen1.5-0.5b-chat":
        filtered_completions = [
            completion[:completion.find("<|im_end|>")] for completion in completions
        ]
    elif model_name == "mistral-7B-instruct":
        filtered_completions = [
            completion[:completion.find("</s>")] for completion in completions
        ]
    elif model_name == "gemma-2b-it":
        filtered_completions = [
            completion[:completion.find("<eos>")] for completion in completions
        ]
    else:
        raise ValueError(f"Model {model_name} not supported. Add to the code.")
    return instructions, filtered_completions

# %%

# Get activations corresponding to loss mask
def get_acts(chat_model, dataset_iterator, hook_point, hook_layer):
    acts_list = []
    for inputs, loss_mask in dataset_iterator:
        input_ids = inputs["input_ids"]
        
        _, cache  = chat_model.run_with_cache(
            input_ids,
            return_type=None,
            names_filter=hook_point,
            stop_at_layer=hook_layer + 1,
        )
        
        resid_acts = cache[hook_point]
        indices = torch.nonzero(loss_mask, as_tuple=True)
        acts = resid_acts[indices[0], indices[1]]
        acts_list.append(acts)

    return torch.cat(acts_list, dim=0)
# %%
# Recons loss
def get_loss_with_mask(model, input_ids, loss_mask, fwd_hooks=[], accumulate=False, debug=False):

    input_ids = input_ids.to(model.cfg.device)
    logits = model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_for_labels = log_probs[:, :-1].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # add a last column of zeros to log_probs_for_labels to match the shape of loss_mask
    log_probs_for_labels = torch.cat(
        [
            log_probs_for_labels,
            torch.zeros(log_probs_for_labels.shape[0]).unsqueeze(-1).to(log_probs_for_labels)
        ],
        dim=-1
    )
    
    log_probs_for_labels = log_probs_for_labels * loss_mask.to(log_probs_for_labels.device)
    if debug:
        print(log_probs_for_labels.shape)
        lines([-log_probs_for_labels[0], loss_mask[0]],
              xaxis="token",
              x=[f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(input_ids))],
              title=model.to_string(input_ids[0]),
        )
        print(model.to_string(input_ids[0]))
    accumulated_loss = -log_probs_for_labels.sum()
    accumulated_n_tokens = loss_mask.sum()
    
    ce_loss = accumulated_loss / accumulated_n_tokens
    
    if accumulate:
        return accumulated_loss, accumulated_n_tokens

    return ce_loss

# %%
@torch.no_grad()
def get_recons_loss_multiple_batches(
    sae: SAE,
    model: HookedRootModule,
    data_iterator,
    scale_activations: bool,
    estimated_norm_scaling_factor: float,
    ignore_outliers: bool = False,
    outlier_threshold: int = 100_000
):
    hook_name = sae.cfg.hook_name
    def standard_replacement_hook(activations: torch.Tensor, hook: Any, loss_mask: torch.Tensor):

        original_device = activations.device
        activations = activations.to(sae.device)
    
        # Handle rescaling if SAE expects it
        if scale_activations:
            activations *= estimated_norm_scaling_factor

        # SAE class agnost forward forward pass. 
        recons_acts = sae.decode(sae.encode(activations)).to(activations.dtype)       
        if ignore_outliers:
            outlier_norm_mask = activations.norm(dim=-1) > outlier_threshold
            mask = loss_mask.bool().unsqueeze(-1).to(recons_acts.device)
            mask = mask & ~outlier_norm_mask.unsqueeze(-1)
            activations = torch.where(mask, recons_acts, activations)
        else:
            mask = loss_mask.bool().unsqueeze(-1).to(recons_acts.device)
            activations = torch.where(mask, recons_acts, activations)

        # Unscale if activations were scaled prior to going into the SAE
        if scale_activations:
            activations /= estimated_norm_scaling_factor
        
        return activations.to(original_device)

    def zero_ablate_hook(activations: torch.Tensor, hook: Any, loss_mask: torch.Tensor):
        original_device = activations.device
        activations = activations.to(sae.device)
        
        # Use broadcasting to apply the mask
        mask = loss_mask.bool().unsqueeze(-1).to(activations.device)  # Shape: [1, 88, 1]
        zero_tensor = torch.zeros_like(activations)
        
        # Use torch.where to selectively zero out activations
        activations = torch.where(mask, zero_tensor, activations)
        
        return activations.to(original_device)
    
    replacement_hook = standard_replacement_hook

    accumulated_clean_loss = 0
    accumulated_recons_loss = 0
    accumulated_zero_abl_loss = 0
    accumulated_tokens = 0
    # for inputs, loss_mask in data_iterator:
    for inputs, loss_mask in tqdm(data_iterator):
        batch_tokens = inputs["input_ids"]
        acc_clean_loss, acc_clean_tokens = get_loss_with_mask(model, batch_tokens, loss_mask, accumulate=True, debug=False)
        accumulated_clean_loss += acc_clean_loss
        
        acc_recons_loss, acc_recons_tokens = get_loss_with_mask(model, batch_tokens, loss_mask, fwd_hooks=[(hook_name, partial(replacement_hook, loss_mask=loss_mask))], accumulate=True, debug=False)
        accumulated_recons_loss += acc_recons_loss
        
        acc_zero_abl_loss, acc_zero_abl_tokens = get_loss_with_mask(model, batch_tokens, loss_mask, fwd_hooks=[(hook_name, partial(zero_ablate_hook, loss_mask=loss_mask))], accumulate=True)
        accumulated_zero_abl_loss += acc_zero_abl_loss
        
        assert acc_clean_tokens == acc_recons_tokens and acc_clean_tokens == acc_zero_abl_tokens
        accumulated_tokens += acc_clean_tokens


    loss = accumulated_clean_loss / accumulated_tokens
    zero_abl_loss = accumulated_zero_abl_loss / accumulated_tokens
    recons_loss = accumulated_recons_loss / accumulated_tokens
    
    div_val = zero_abl_loss - loss
    div_val[torch.abs(div_val) < 0.0001] = 1.0

    score = (zero_abl_loss - recons_loss) / div_val
    return score, loss, recons_loss, zero_abl_loss

# %%
def get_or_generate_rollouts(model, tokenize_instructions_fn, filename='generated_data.pkl', num_samples=100):
    if os.path.exists(filename):
        print(f"Loading existing data from '{filename}'")
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        instructions = loaded_data['instructions']
        completions = loaded_data['completions']
    else:
        print(f"Generating new data...")
        instructions, completions = generate_rollouts(model, tokenize_instructions_fn, num_samples=num_samples)
        
        data_to_save = {
            'instructions': instructions,
            'completions': completions
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Data saved to '{filename}'")

    return instructions, completions
# %%

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--chat_model_name',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--base_sae_project',
        help='Name of base SAE project from wandb')
    parser.add_argument(
        '--chat_sae_project',
        help='Name of chat SAE project from wandb')
    parser.add_argument(
        '--wandb_entity',
        help='Name of wandb entity wandb', default="ckkissane")
    parser.add_argument(
        '--mask_type',
        help='Type of mask to apply for evals (rollout, user_prompt, control_tokens)')
    parser.add_argument(
        '--num_samples',
        help='Number of samples from the alpaca dataset', type=int)
    parser.add_argument(
        '--ignore_outliers', action='store_true',
        help='whether to ignore outlier norms when calculating recons loss')
    parser.add_argument(
        '--batch_size',
        help='batch size for evals', type=int)
    parser.add_argument(
        '--outlier_threshold',
        help='threshold for outlier acts to ignore', type=int)
    
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)
    
    
    args = parser.parse_args()
    assert args.chat_model_name in ["qwen1.5-0.5b-chat", "mistral-7B-instruct", "gemma-2b-it"], "Only qwen1.5 0.5 chat, mistral 7B instruct, and gemma-2b-it it are supported, change the code"
    assert args.mask_type in ["rollout", "user_prompt", "control_tokens"], "Invalid mask type, choose from rollout, user_prompt, control_tokens"
    chat_model = HookedTransformer.from_pretrained(args.chat_model_name)
    
    if args.ignore_outliers:
        print(f"Ignoring outliers. outlier_threshold: {args.outlier_threshold}")

    if args.chat_model_name == "qwen1.5-0.5b-chat":
        data_file = 'generated_data_qwen.pkl'
    elif args.chat_model_name == "mistral-7B-instruct":
        data_file = 'generated_data_mistral.pkl'
    elif args.chat_model_name == "gemma-2b-it":
        data_file = 'generated_data_gemma.pkl'
    else:
        raise ValueError("Model not supported, add to code")
    
    if args.chat_model_name == "qwen1.5-0.5b-chat":
        soi_toks, eoi_toks = chat_model.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[0]), chat_model.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])
        tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=chat_model.tokenizer, system=None, include_trailing_whitespace=True)
    elif args.chat_model_name == "mistral-7B-instruct":
        soi_toks, eoi_toks = chat_model.tokenizer.encode(MISTRAL_CHAT_TEMPLATE.split("{instruction}")[0], add_special_tokens=True), chat_model.tokenizer.encode(MISTRAL_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)
        soi_toks = soi_toks[:-1] # tokenization hack
        eoi_toks = eoi_toks[:-1] # tokenization hack
        tokenize_instructions_fn = functools.partial(tokenize_instructions_mistral_chat, tokenizer=chat_model.tokenizer, include_trailing_whitespace=True)
    elif args.chat_model_name == "gemma-2b-it":
        soi_toks, eoi_toks = chat_model.tokenizer.encode(GEMMA_CHAT_TEMPLATE.split("{instruction}")[0], add_special_tokens=False), chat_model.tokenizer.encode(GEMMA_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)
        tokenize_instructions_fn = functools.partial(tokenize_instructions_gemma_chat, tokenizer=chat_model.tokenizer, include_trailing_whitespace=True)
    else:
        raise ValueError("Only qwen1.5-0.5b-chat, mistral-7B-instruct, and gemma-2b-it are supported, change the code")

    instructions, completions = get_or_generate_rollouts(chat_model, tokenize_instructions_fn, filename=data_file, num_samples=args.num_samples)
    instructions, completions = filter_completions(args.chat_model_name, instructions=instructions, completions=completions)

    # Load SAEs
    entity = args.wandb_entity
    
    assert args.chat_sae_project in ["qwen-500M-chat", "mistral-7B-chat", "gemma-2b-it"], "only support qwen1.5 0.5B / mistral 7B / gemma 2B chat for now, change the code"
    assert args.base_sae_project in ["qwen-500M-base", "mistral-7B-base", "gemma-2b-base"], "only support qwen1.5 0.5B / mistral 7B / gemma 2B base for now, change the code"
    projects = [args.chat_sae_project, args.base_sae_project]

    for project in projects:
        if project == "qwen-500M-chat":
            artifact = "sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768:v0"
            chat_path = f"./artifacts/{artifact}"
        elif project == "qwen-500M-base":
            artifact = "sae_qwen1.5-0.5b_blocks.13.hook_resid_pre_32768:v0"
            base_path = f"./artifacts/{artifact}"
        elif project == "mistral-7B-chat":
            artifact = artifact = "sae_mistral-7b-instruct_blocks.16.hook_resid_pre_131072:v8"
            chat_path = f"./artifacts/{artifact}"
        elif project == "mistral-7B-base":
            artifact = "sae_mistral-7b_blocks.16.hook_resid_pre_131072:v8"
            base_path = f"./artifacts/{artifact}"
        elif project == "gemma-2b-it":
            project = "gemma-2"
            artifact = "sae_gemma-2b-it_blocks.9.hook_resid_pre_32768:v7"
            chat_path = f"./artifacts/{artifact}"
        elif project == "gemma-2b-base":
            project = "gemma-2"
            artifact = "sae_gemma-2b_blocks.9.hook_resid_pre_32768:v8"
            base_path = f"./artifacts/{artifact}"
        else:
            raise ValueError(f"Project {project} Not supported, add to the code.")
        artifact_path = f"{entity}/{project}/{artifact}"
        api = wandb.Api()
        artifact = api.artifact(artifact_path)
        artifact.download()

    chat_sae = SAE.load_from_pretrained(chat_path, device=device)
    base_sae = SAE.load_from_pretrained(base_path, device=device)
    
    # Get activations corresponding to mask_type
    dataset_iterator = batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, args.batch_size, eoi_toks=torch.tensor(eoi_toks), soi_toks=torch.tensor(soi_toks), mask_type=args.mask_type)
    all_activations = get_acts(chat_model, dataset_iterator, chat_sae.cfg.hook_name, chat_sae.cfg.hook_layer)
    
    if args.chat_model_name == "qwen1.5-0.5b-chat":
        chat_norm_scaling_factor = 1.573157268076237
    elif args.chat_model_name == "mistral-7B-instruct":
        chat_norm_scaling_factor = 7.931667509979551
    elif args.chat_model_name == "gemma-2b-it":
        chat_norm_scaling_factor = 2.419691849168719
    else:
        raise ValueError("Model not supported, add to code")
    
    all_activations *= chat_norm_scaling_factor

    print(f"Evaluating MSE / L0 / EV on {all_activations.shape[0]} activations")
    # MSE
    chat_mse = get_mse_from_batch(chat_sae, all_activations, args.ignore_outliers, outlier_threshold=args.outlier_threshold)
    print("chat mse", chat_mse)

    base_mse = get_mse_from_batch(base_sae, all_activations, args.ignore_outliers, outlier_threshold=args.outlier_threshold)
    print("base mse", base_mse)

    # L0
    chat_l0 = get_l0_from_batch(chat_sae, all_activations, args.ignore_outliers, outlier_threshold=args.outlier_threshold)
    print("chat l0", chat_l0)

    base_l0 = get_l0_from_batch(base_sae, all_activations, args.ignore_outliers, outlier_threshold=args.outlier_threshold)
    print("base l0", base_l0)

    # Explained variance
    chat_ev = get_explained_variance_from_batch(chat_sae, all_activations, args.ignore_outliers, outlier_threshold=args.outlier_threshold)
    print("chat explained variance", chat_ev.mean().item())

    base_ev = get_explained_variance_from_batch(base_sae, all_activations, args.ignore_outliers, outlier_threshold=args.outlier_threshold)
    print("base explained variance", base_ev.mean().item())
    
    # CE evals
    
    # Splice Chat SAE -> Chat Model
    dataset_iterator = batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, args.batch_size, eoi_toks=torch.tensor(eoi_toks), soi_toks=torch.tensor(soi_toks), mask_type=args.mask_type)

    chat_chat_ce_rec, chat_clean_loss, chat_chat_sae_loss, zero_abl_loss = get_recons_loss_multiple_batches(
        chat_sae,
        chat_model,
        dataset_iterator,
        scale_activations=True,
        estimated_norm_scaling_factor=chat_norm_scaling_factor,
        ignore_outliers = args.ignore_outliers,
        outlier_threshold=args.outlier_threshold,
    )
    print(f"chat_chat_ce_rec: {chat_chat_ce_rec}")
    print(f"chat_chat_sae_loss: {chat_chat_sae_loss}")
    
    # Splice Base SAE -> Chat Model
    dataset_iterator = batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, args.batch_size, eoi_toks=torch.tensor(eoi_toks), soi_toks=torch.tensor(soi_toks), mask_type=args.mask_type)

    base_chat_ce_rec, chat_clean_loss, base_chat_sae_loss, zero_abl_loss = get_recons_loss_multiple_batches(
        base_sae,
        chat_model,
        dataset_iterator,
        scale_activations=True,
        estimated_norm_scaling_factor=chat_norm_scaling_factor,
        ignore_outliers = args.ignore_outliers,
        outlier_threshold=args.outlier_threshold,
    )

    print(f"base_chat_ce_rec: {base_chat_ce_rec}")
    print(f"chat_clean_loss: {chat_clean_loss}")
    print(f"base_chat_sae_loss: {base_chat_sae_loss}")
    print(f"zero_abl_loss: {zero_abl_loss}")
    
    records = {
        "chat_model": args.chat_model_name,
        "mask_type": args.mask_type,
        "chat_mse": chat_mse.mean().item(),
        "base_mse": base_mse.mean().item(),
        "chat_l0": chat_l0.mean().item(),
        "base_l0": base_l0.mean().item(),
        "chat_ev": chat_ev.mean().item(),
        "base_ev": base_ev.mean().item(),
        "Base->Chat CE Rec": base_chat_ce_rec.mean().item(),
        "Chat->Chat CE Rec": chat_chat_ce_rec.mean().item(),
        "Clean Chat loss": chat_clean_loss.mean().item(),
        "Chat->Chat SAE loss": chat_chat_sae_loss.mean().item(),
        "Base->Chat SAE loss": base_chat_sae_loss.mean().item(),
        "Zero abl loss": zero_abl_loss.mean().item(),
        "ignore_outliers": args.ignore_outliers,
        "outlier_threshold": args.outlier_threshold,
    }

    results_file = "./results/alpaca_sae_transfer_results.json"

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

# %%