import numpy as np

def extract_xml_answer(text: str) -> str:
    answer = text.split("<genes>")[-1]
    answer = answer.split("</genes>")[0]
    return answer.strip()

def extract_xml_reasoning(text: str) -> str:
    answer = text.split("<reasoning>")[-1]
    answer = answer.split("</reasoning>")[0]
    return answer.strip()

def get_gene_list(response):
    ans = extract_xml_answer(response).replace("[","").replace("]","")
    genes = [x.strip() for x in ans.split(",")]
    return genes

def binary_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [get_gene_list(r) for r in responses]
    return [1.0 if len(r) >=1 and a==r[0] else 0.0 for r, a in zip(extracted_responses, answer)]

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [get_gene_list(r) for r in responses]
    return [1.0/(1+r.index(a)) if a in r else 0.0 for r, a in zip(extracted_responses, answer)]

def get_system_prompt():
    SYSTEM_PROMPT = "Your task is to identify likely causal genes for a given GWAS phenotype."
    SYSTEM_PROMPT += "You will be given the name of a GWAS phenotype and a list of candidate genes, "
    SYSTEM_PROMPT += "and you must respond with genes from the list that may be responsible. "
    SYSTEM_PROMPT += "The genes should be sorted in the order of most-confident to least-confident, "
    SYSTEM_PROMPT += "with the most confident first and the least confident last.\n"
    SYSTEM_PROMPT += "Respond in the following format:\n"
    SYSTEM_PROMPT += """ <reasoning>
    reasoning about the phenotype, the types of genes potentially involved, and the function of each provided gene.
    </reasoning>
    <genes>
    [<gene_1>, <gene_2>, ... <gene_N>]
    </genes>

    Start your response with <reasoning>.
    """
    return SYSTEM_PROMPT

def get_system_prompt_short_reasoning():
    SYSTEM_PROMPT = "Your task is to identify likely causal genes for a given GWAS phenotype."
    SYSTEM_PROMPT += "You will be given the name of a GWAS phenotype and a list of candidate genes, "
    SYSTEM_PROMPT += "and you must respond with genes from the list that may be responsible. "
    SYSTEM_PROMPT += "The genes should be sorted in the order of most-confident to least-confident, "
    SYSTEM_PROMPT += "with the most confident first and the least confident last.\n"
    SYSTEM_PROMPT += "Respond in the following format:\n"
    SYSTEM_PROMPT += """ <reasoning>
    reasoning about the phenotype, the types of genes potentially involved, and the function of each provided gene. Use 200 words or less.
    </reasoning>
    <genes>
    [<gene_1>, <gene_2>, ... <gene_N>]
    </genes>

    Start your response with <reasoning>.
    """
    return SYSTEM_PROMPT

def get_system_prompt_short_reasoning_v2():
    SYSTEM_PROMPT = "Your task is to identify likely causal genes for a given GWAS phenotype."
    SYSTEM_PROMPT += "You will be given the name of a GWAS phenotype and a list of candidate genes, "
    SYSTEM_PROMPT += "and you must respond with genes from the list that may be responsible. "
    SYSTEM_PROMPT += "The genes should be sorted in the order of most-confident to least-confident, "
    SYSTEM_PROMPT += "with the most confident first and the least confident last.\n"
    SYSTEM_PROMPT += "Respond in the following format:\n"
    SYSTEM_PROMPT += """ <reasoning>
    reasoning about the phenotype, the types of genes potentially involved, and the function of each provided gene. Keep your reasoning short.
    </reasoning>
    <genes>
    [<gene_1>, <gene_2>, ... <gene_N>]
    </genes>

    Start your response with <reasoning>.
    """
    return SYSTEM_PROMPT

def prompt(trait,gene_str):
    query_str = f"List all likely causal genes.\nGWAS phenotype: {trait}\n"
    query_str += f"Genes in locus: {gene_str}"
    return query_str

