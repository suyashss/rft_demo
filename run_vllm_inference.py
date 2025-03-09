from pathlib import Path
from utils import * 
import os

import modal

app = modal.App("rft-inference")

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("rft-demo-vol", create_if_missing=True)
MODEL_DIR = Path("/root/models")
GPU_CONFIG = "A100-80GB:2"

# define dependencies for downloading model
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.7.3","datasets","pandas",
                 gpu=GPU_CONFIG.split(":")[0])
    .add_local_python_source("utils")
)

def setup_inference_dataset(hf_dataset_name: str, hf_dataset_config: str):
    from datasets import load_dataset, Dataset

    # Load dataset
    ds = load_dataset(hf_dataset_name, hf_dataset_config,split='eval')

    ds = ds.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': get_system_prompt()},
            {'role': 'user', 'content': prompt(x['description'], x['symbol_gene_string'])}
        ],
        'answer': x['symbol']
    })

    return ds


@app.function(
    volumes={MODEL_DIR: volume},  # "mount" the Volume, sharing it with your function
    image=inference_image,  # only download dependencies needed here
    timeout=3600*2, # set longer timeout for large model download
    gpu=GPU_CONFIG
)
def launch(
    model_name: str, output_dir: str, output_fname: str,
    hf_dataset_name: str, hf_dataset_config: str = 'default',
    lora_path: str = None, lora_rank: int = 32, 
    max_tokens: int = 512, batch_size: int = 256,
    distribute_inference: bool = False
    ):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import pandas as pd

    model_path = str(MODEL_DIR/model_name)
    
    sampling_params = SamplingParams(temperature=0.01,max_tokens=max_tokens,seed=1)
    if lora_path:
        lora_path = str(MODEL_DIR/lora_path)
        llm = LLM(model=model_path,enable_lora=True,max_lora_rank=lora_rank,
                  gpu_memory_utilization=0.95,max_model_len=2048)
    elif distribute_inference:
        NUM_GPUS = int(GPU_CONFIG.split(":")[1])
        llm = LLM(model=model_path,gpu_memory_utilization=0.95,max_model_len=2048,tensor_parallel_size=NUM_GPUS)
    else:
        llm = LLM(model=model_path,gpu_memory_utilization=0.95,max_model_len=2048)
        
    dataset = setup_inference_dataset(hf_dataset_name, hf_dataset_config)

    results_dir = MODEL_DIR / output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    result_list = []
    for i in range(0,len(dataset),batch_size):
        batch_start, batch_end = i, min(i + batch_size, len(dataset))
        batch = dataset.select(range(batch_start,batch_end))
        prompts = [example['prompt'] for example in batch]
        true_answers = [example['answer'] for example in batch]

        if lora_path:
            outputs = llm.chat(prompts,sampling_params=sampling_params,use_tqdm=True,
                               lora_request=LoRARequest("lora",1,lora_path))
        else:
            outputs = llm.chat(prompts,sampling_params=sampling_params,use_tqdm=True)

        for j,output in enumerate(outputs):
            gen_text = output.outputs[0].text
            ans = {
                'reasoning': extract_xml_reasoning(gen_text),
                'genes': extract_xml_answer(gen_text),
                'raw_response': gen_text,
                'true_answer': true_answers[j]
            }
            result_list.append(ans)

    result_df = pd.DataFrame(result_list)
    result_df.to_csv(str(results_dir/output_fname))
    print("Done")

@app.local_entrypoint()
def main(
    model_name:str, output_dir: str, output_fname: str, 
    hf_dataset_name: str, hf_dataset_config: str = 'default',
    lora_path: str = None, lora_rank: int = 32,
    max_tokens: int = 512, batch_size: int = 256,
    distribute_inference: bool = False
         ):
    launch.remote(model_name,output_dir, output_fname,
                  hf_dataset_name,hf_dataset_config,
                  lora_path,lora_rank,max_tokens,batch_size,distribute_inference)

