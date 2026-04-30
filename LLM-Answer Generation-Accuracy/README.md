#  LLM Answer Generation Pipeline

This repository provides a **evaluation pipeline** for benchmarking **large language models (LLMs)** on **different tasks**. There is a separate file for code generation, mathematical reasoning, question answering and text summarization. 


The scripts evaluate multiple pretrained LLMs on a different datasets and produces **model-specific CSV outputs** for downstream analysis.

The csv files used in the code can be found in the CSV folder in the main directory.
---

##  Evaluated Models

The script evaluates the following models by default:

```
Llama-2-13b
Yi-34B
qwen2-32B
Deepseek-7b
Falcon3-7B
Llama-2-7b
Llama-3-8B
Phi-3-14B
Qwen2-7B
Yi-6B
gemma-2-9b
Mistral-7B
Qwen2-14B
```

All models must already be downloaded locally (`local_files_only=True`). Moreover, we used instruct models instead of a base one.

---

##  Prompting Strategy

For four different problems, we use four different prompt strategies. The prompt used in these experiments can be found in the respective files.


##  Requirements

### Hardware
- These experiments can only be performed with NVIDIA A100 GPU (80GB) to get the similar results
- CUDA-enabled system

### Software
Software specification is given in the main directory in requirement.txt


##  Running
You can run the file as 
```
python AnswerGeneration-Code.py
```