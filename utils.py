# %%
import os
from IPython import get_ipython

ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import plotly.io as pio
pio.renderers.default = "jupyterlab"

# Import stuff
import einops
import json
from pathlib import Path
import plotly.express as px
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch
import numpy as np
from transformer_lens import HookedTransformer

from functools import partial

from IPython.display import HTML

from transformer_lens.utils import to_numpy
import pandas as pd

from html import escape
import colorsys


import wandb

import plotly.graph_objects as go

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis",
     "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid",
     "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth"
}

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    fig = px.imshow(to_numpy(tensor), color_continuous_midpoint=0.0,labels={"x":xaxis, "y":yaxis}, **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label

    fig.show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(y=to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, return_fig=False, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    if return_fig:
        return fig
    fig.show(renderer)

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def bar(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.bar(
        y=to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        template="simple_white",
        **kwargs).show(renderer)

def create_html(strings, values, saturation=0.5, allow_different_length=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))
    
# SAE Lens stuff
def standard_mse_loss_fn(
    preds: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.mse_loss(preds, target, reduction="none")

mse_loss_fn = standard_mse_loss_fn

def get_mse_from_batch(sae, sae_in, ignore_outliers=False, outlier_threshold=100_000):
    if ignore_outliers:
        outlier_norm_mask = sae_in.norm(dim=-1) > outlier_threshold
        sae_in = sae_in[~outlier_norm_mask]
    sae_out = sae(sae_in)

    # MSE LOSS
    per_item_mse_loss = mse_loss_fn(sae_out, sae_in)
    mse_loss = per_item_mse_loss.sum(dim=-1).mean()
    
    return mse_loss

def get_mse_multiple_batches(
    sae,
    act_store, 
    num_batches,
    ignore_outliers=False,
    outlier_threshold=100_000
):
    mses = torch.zeros(num_batches, device=sae.device)
    for i in range(num_batches):
        sae_in = act_store.next_batch()[:, 0, :].to(sae.device)
        if ignore_outliers:
            outlier_norm_mask = sae_in.norm(dim=-1) > outlier_threshold
            sae_in = sae_in[~outlier_norm_mask]
        get_mse_from_batch(sae, sae_in)
        mses[i] = get_mse_from_batch(sae, sae_in)
    return mses.mean().item()

def get_l0_from_batch(sae, sae_in, ignore_outliers=False, outlier_threshold=100_000):
    if ignore_outliers:
        outlier_norm_mask = sae_in.norm(dim=-1) >outlier_threshold
        sae_in = sae_in[~outlier_norm_mask]
    feature_acts = sae.encode(sae_in)
    did_fire = (feature_acts > 0.0)
    avg_l0 = did_fire.float().sum(dim=-1).mean()
    return avg_l0

def get_l0_multiple_batches(
    sae, 
    act_store, 
    num_batches,
    ignore_outliers=False,
    outlier_threshold=100_000
):
    num_firing = torch.zeros(num_batches, device=sae.device)
    for i in range(num_batches):
        sae_in = act_store.next_batch()[:, 0, :].to(sae.device)
        if ignore_outliers:
            outlier_norm_mask = sae_in.norm(dim=-1) > outlier_threshold
            sae_in = sae_in[~outlier_norm_mask]
        num_firing[i] = get_l0_from_batch(sae, sae_in)
    return num_firing.mean().item()

def get_explained_variance_multiple_batches(
    sae, 
    act_store, 
    num_batches,
    ignore_outliers=False,
    outlier_threshold=100_000
):
    explained_variances = torch.zeros(num_batches, device=sae.device)
    for i in range(num_batches):
        sae_in = act_store.next_batch()[:, 0, :].to(sae.device)
        if ignore_outliers:
            outlier_norm_mask = sae_in.norm(dim=-1) > outlier_threshold
            sae_in = sae_in[~outlier_norm_mask]
        explained_variances[i] = get_explained_variance_from_batch(sae, sae_in).mean().item()
    return explained_variances.mean().item()

def get_explained_variance_from_batch(sae, sae_in, ignore_outliers=False, outlier_threshold=100_000):
    if ignore_outliers:
        outlier_norm_mask = sae_in.norm(dim=-1) > outlier_threshold
        sae_in = sae_in[~outlier_norm_mask]
    sae_out = sae(sae_in)
    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
    total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
    explained_variance = 1 - per_token_l2_loss / total_variance
    return explained_variance

# Chat formatting stuff (heavily cribbed from Andy's code)
import functools
from typing import List
from transformers import AutoTokenizer

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction



def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

MISTRAL_CHAT_TEMPLATE = """ [INST] {instruction} [/INST] """

def format_instruction_mistral_chat(
    instruction: str,
    output: str=None,
    include_trailing_whitespace: bool=True,
):

    formatted_instruction = MISTRAL_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_mistral_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_mistral_chat(instruction=instruction, output=output, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_mistral_chat(instruction=instruction, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

def format_instruction_gemma_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")
    else:
        formatted_instruction = GEMMA_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

from jaxtyping import Int
from torch import Tensor
from typing import List

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=False)
# %%
