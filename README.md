# Recreate the reinforcement fine-tuning demo

Steps 

1. Data generation (skip if you want to directly download from Huggingface): 
* Download our supplementary files from zenodo using `download.sh`. 
* Run `python setup_datasets.py --loci_files ./zenodo_directory/data/benchmark_datasets/pharmaprojects_step2.for_llm.tsv ./zenodo_directory/data/benchmark_datasets/gwas_catalog_step2.for_llm.tsv  --label_files ./zenodo_directory/data/benchmark_datasets/pharmaprojects_step2.labels ./zenodo_directory/data/benchmark_datasets/gwas_catalog_step2.labels --hf_dataset <YOUR_DATASET> --hf_dataset_config default`. This will create the dataset and upload it to Huggingface
2. Setup modal run by downloading your models. 
`python setup_models.py`
3. Run GRPO fine-tuning on modal.
`python -m modal run run_grpo.py --detach --model_name meta-llama/Llama-3.1-8B-Instruct --hf_dataset_name suyashss/gwas_genes output_dir results`
4. Run inference and copy files to local folder.
`bash inference.sh`
5. Evaluation: use the `evaluation.ipynb` juypter notebook.