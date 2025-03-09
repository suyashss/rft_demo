from pathlib import Path
from utils import * 
import os

import modal

app = modal.App("rft-demo-model-setup")

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("rft-demo-vol", create_if_missing=True)
MODEL_DIR = Path("/root/models")
GPU_CONFIG = "H100:1"

# define dependencies for downloading model
rft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("unsloth==2025.2.15","vllm==0.7.3","trl==0.15.1","pillow==11.1.0",
                 "wandb",gpu=GPU_CONFIG.split(":")[0])
    .env({"WANDB_PROJECT": "rft_demo"})
    .add_local_python_source("utils")
)

def setup_model(model_name: str, lora_rank: int = 32):
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    import torch
    max_seq_length = 1024 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(MODEL_DIR/model_name),
        max_seq_length = max_seq_length,
        load_in_4bit = True, 
        fast_inference = True, 
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, 
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    return model, tokenizer

def setup_dataset(hf_dataset_name: str, hf_dataset_config: str):
    from datasets import load_dataset, Dataset

    # Load dataset
    ds = load_dataset(hf_dataset_name, hf_dataset_config,split='train')

    ds = ds.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': get_system_prompt()},
            {'role': 'user', 'content': prompt(x['description'], x['symbol_gene_string'])}
        ],
        'answer': x['symbol']
    })
    #print(ds)
    split_ds = ds.train_test_split(test_size=0.1,shuffle=False)
    train_ds,eval_ds = split_ds['train'],split_ds['test'] 

    return train_ds,eval_ds

def setup_training_args(output_dir: str):
    from trl import GRPOConfig
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        vllm_max_model_len = 700,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.01,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 128,
        per_device_eval_batch_size = 128,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 16, # Decrease if out of memory
        max_prompt_length = 256,
        max_completion_length = 500,
        num_train_epochs = 10, # Set to 1 for a full training run
        save_steps = 100,
        eval_steps = 100,
        eval_strategy = 'steps',
        max_grad_norm = 0.1,
        beta = 0.01,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = output_dir,
        log_completions = True
    )
    return training_args

@app.function(
    volumes={MODEL_DIR: volume},  # "mount" the Volume, sharing it with your function
    image=rft_image,  # only download dependencies needed here
    secrets=[modal.Secret.from_name("wandb")], # huggingface token
    timeout=3600*24, # set longer timeout for training
    gpu=GPU_CONFIG
)
def launch(
    model_name: str, hf_dataset_name: str, output_dir: str, hf_dataset_config: str = 'default',
    lora_rank: int = 32
    ):

    model, tokenizer = setup_model(model_name, lora_rank)
    
    train_ds, eval_ds = setup_dataset(hf_dataset_name, hf_dataset_config)

    results_dir = MODEL_DIR / output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    training_args = setup_training_args(str(results_dir))

    from trl import GRPOTrainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [correctness_reward_func],
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
    )
    trainer.train()


@app.local_entrypoint()
def main(model_name:str, 
         hf_dataset_name: str,
         output_dir: str, 
         hf_dataset_config: str = 'default'):
    launch.remote(model_name,hf_dataset_name,output_dir,hf_dataset_config)
