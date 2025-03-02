from pathlib import Path
import os

import modal

app = modal.App("rft-demo-model-setup")

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("rft-demo-vol", create_if_missing=True)
MODEL_DIR = Path("/root/models")

# define dependencies for downloading model
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")  # install fast Rust download client
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
)

# define dependencies for running model
#inference_image =  modal.Image.debian_slim().pip_install("transformers")

@app.function(
    volumes={MODEL_DIR: volume},  # "mount" the Volume, sharing it with your function
    image=download_image,  # only download dependencies needed here
    secrets=[modal.Secret.from_name("huggingface")] # huggingface token
)
def download_model(
    repo_id: str="hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    revision: str=None,  # include a revision to prevent surprises!
    ):
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR / repo_id, 
                      token = os.environ["HF_TOKEN"], ignore_patterns = "*.pth")
    print(f"Model downloaded to {MODEL_DIR / repo_id}")


@app.local_entrypoint()
def main():
    model1 = "meta-llama/Llama-3.1-70B-Instruct"
    download_model.remote(model1)

#    model2 = "meta-llama/Llama-3.1-8B-Instruct"
#    download_model.remote(model2)
#    print("Downloaded all models")

