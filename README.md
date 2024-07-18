Code to reproduce key results accompanying "SAEs (usually) Transfer Between Base and Chat Models".

* [Blog post](https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models)

## Contents

* `sae_transfer_script.py` contains code to reproduce SAE transfer evaluations on the pile. Example usage:

`python sae_transfer_script.py --sae_project mistral-7B-base --model_name mistral-7B-instruct --wandb_entity ckkissane --eval_batch_size_prompts 8 --eval_batches 6 --num_act_batches 10 --ignore_outliers --outlier_threshold 200`

* `alpaca_transfer_script.py` contains code to reproduce SAE transfer evaluations on the alpaca. Example usage:

`python alpaca_transfer_script.py --chat_model_name mistral-7B-instruct --base_sae_project mistral-7B-base --chat_sae_project mistral-7B-chat --mask_type rollout --num_samples 100 --ignore_outliers --batch_size 1 --outlier_threshold 200`

* `finetuned_sae_evals.py` contains code to evaluate the fine-tuned SAEs. Example usage:

`python finetuned_sae_evals.py  --model_name mistral-7B-instruct --wandb_entity ckkissane --eval_batch_size_prompts 8 --eval_batches 6 --num_act_batches 10`

* `generated_data_*.pkl` contain ~50 alpaca instructions / completions generated from each of Mistral-7B Instruct, Gemma v1 2B IT, and Qwen 1.5 0.5B Chat.

We build on code from [SAELens](https://github.com/jbloomAus/SAELens) and [Arditi et al.](https://github.com/andyrdt/refusal_direction)

## Open source SAEs

We open source SAEs used in this work:

* [Mistral 7B base](https://wandb.ai/ckkissane/mistral-7B-base/artifacts/model/sae_mistral-7b_blocks.16.hook_resid_pre_131072/v8/files), [Mistral 7B chat](https://wandb.ai/ckkissane/mistral-7B-chat/artifacts/model/sae_mistral-7b-instruct_blocks.16.hook_resid_pre_131072/v8/files), [Mistral 7B fine-tuned](https://wandb.ai/ckkissane/mistral-7B-base-finetuning/artifacts/model/sae_mistral-7b_blocks.16.hook_resid_pre_131072/v1/files)
* [Qwen 1.5 0.5B base](https://wandb.ai/ckkissane/qwen-500M-base/artifacts/model/sae_qwen1.5-0.5b_blocks.13.hook_resid_pre_32768/v0/files), [Qwen 1.5 0.5B chat](https://wandb.ai/ckkissane/qwen-500M-chat/artifacts/model/sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768/v0/files), [Qwen 1.5 0.5B fine-tuned](https://wandb.ai/ckkissane/qwen-500M-base-finetuning/artifacts/model/sae_qwen1.5-0.5b_blocks.13.hook_resid_pre_32768/v6/files)
