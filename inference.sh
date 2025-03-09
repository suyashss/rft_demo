modal run --detach run_vllm_inference.py::main --model-name meta-llama/Llama-3.1-70B-Instruct --hf-dataset-name suyashss/gwas_genes --output-dir inference_results --output-fname zero_shot_70b_results.csv --distribute-inference
modal run --detach run_vllm_inference.py::main --model-name meta-llama/Llama-3.1-8B-Instruct --hf-dataset-name suyashss/gwas_genes --output-dir inference_results --output-fname grpo_8b_results.csv --lora-path results_with_val/checkpoint-700
modal run --detach run_vllm_inference.py::main --model-name meta-llama/Llama-3.1-8B-Instruct --hf-dataset-name suyashss/gwas_genes --output-dir inference_results --output-fname grpo_8b_binary_reward_results.csv --lora-path results_binary_reward/checkpoint-700
modal run --detach run_vllm_inference.py::main --model-name meta-llama/Llama-3.1-8B-Instruct --hf-dataset-name suyashss/gwas_genes --output-dir inference_results --output-fname zero_shot_8b_results.csv 
modal volume get rft-demo-vol inference_results/grpo_8b_results.csv --force
modal volume get rft-demo-vol inference_results/grpo_8b_binary_reward_results.csv --force
modal volume get rft-demo-vol inference_results/zero_shot_8b_results.csv --force
modal volume get rft-demo-vol inference_results/zero_shot_70b_results.csv --force
mv *results.csv results/