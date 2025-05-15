# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune `meta_llama_3_8b` model from marketplace with QLORA
# MAGIC
# MAGIC Databricks hosts the `meta_llama_3_8b` in [Databricks marketplace](https://marketplace.databricks.com/). This is a tutorial to show how to fine tune the models on the [mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.1 GPU ML Runtime
# MAGIC - Instance: `g5.8xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure, `g2-standard-8` or `a2-highgpu-1g` on GCP
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient fine tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# To access models in Unity Catalog, ensure that MLflow is up to date
!pip install git+https://github.com/huggingface/peft
!pip install git+https://github.com/huggingface/accelerate
!pip install torchvision
!pip install bitsandbytes==0.43.2
!pip install -U trl
!pip install -U transformers
dbutils.library.restartPython()

# COMMAND ----------

import os
dbutils.widgets.text("huggingface_token", "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", "Enter Parameter")
# ! export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] ='max_split_size_mb:128'

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

torch.cuda.memory_summary() 

# COMMAND ----------

import mlflow
# import torch
torch.cuda.empty_cache()
# Set mlflow registry to databricks-uc
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the model
# MAGIC
# MAGIC In this section we will load the [meta_llama_3_8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model installed from [Databricks marketplace](https://marketplace.databricks.com/) saved in Unity Catalog to local disk, quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

# from huggingface_hub import notebook_login
# notebook_login()
import os

os.environ['HF_TOKEN'] = dbutils.widgets.get("huggingface_token")

# COMMAND ----------

# MAGIC %md
# MAGIC Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. 

# COMMAND ----------

from peft import LoraConfig
linear_layers = ['gate_proj', 'v_proj', 'o_proj', 'k_proj', 'up_proj', 'down_proj', 'q_proj']
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=linear_layers,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Then finally pass everthing to the trainer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the LORA model

# COMMAND ----------

# trainer.save_model('/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/meta_llama_3_8b_lora_fine_tune')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the fine tuned model to MLFlow

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import mlflow

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Specify an input example that conforms to the input schema for the task.
import numpy as np
input_example={"prompt": np.array(["Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"]),
                                   "temperature": 0.5,
                                   "max_tokens": 100}
        


# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge and save as standalone model

# COMMAND ----------

# MAGIC %md
# MAGIC Load model base model in 16 or 32 bits (full precision) and merge with the PEFT adapter

# COMMAND ----------

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from peft import PeftModel

# base_model_id = 'meta-llama/Llama-3.1-8B'
# base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cuda:0",torch_dtype=torch.float16)
# peft_model = PeftModel.from_pretrained(base_model, '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/meta_llama_3_8b_lora_fine_tune')
# tokenizer = AutoTokenizer.from_pretrained(base_model_id)


# COMMAND ----------

# merged_model = peft_model.merge_and_unload()
# merged_model.eval()
# merged_model.save_pretrained(merged_model_path)
# tokenizer.save_pretrained(merged_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Finetuned Standalone model

# COMMAND ----------

# MAGIC %md
# MAGIC Load model for inferenece (could be quantized)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformer Flavor

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

merged_model_path = '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/merged_meta_llama_3_8b_lora_fine_tune'
# merged_model= AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16, device_map="cudaf:0")
merged_model= AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# COMMAND ----------

# DBTITLE 1,Load Model in 4 bits
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import torch
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        )

merged_model_path = '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/merged_meta_llama_3_8b_lora_fine_tune'
merged_model= AutoModelForCausalLM.from_pretrained(merged_model_path, load_in_4bit=True).to("cuda")
# merged_model= AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# COMMAND ----------

# Use model in eval mode to log model optimzed for inference (e.g., no dropout layers)
merged_model.eval()

# COMMAND ----------


catalog = "cindy_demo_catalog"
schema = "llm_fine_tuning"
model_name = "meta_llama_3_8b_lora_fine_tuned_4bits"
registered_model_name = f"{catalog}.{schema}.{model_name}"

with mlflow.start_run():
    mlflow.log_params(peft_config.to_dict())
    
    components = {
      "model": merged_model,
      "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        # Specify the llm/v1/xxx task that is compatible with the model being logged
        task="llm/v1/completions",
        artifact_path="model",# This is a relative path to save model files within MLflow run
        input_example=input_example,
        # By passing the model name, MLflow automatically registers the Transformers model to Unity Catalog with the given catalog/schema/model_name.
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve Registered Model

# COMMAND ----------

import requests
import json

# Set the name of the MLflow endpoint
endpoint_name = "cindy_peft_llama3_8b"

catalog = "cindy_demo_catalog"
schema = "llm_fine_tuning"
model_name = "meta_llama_3_8b_lora_fine_tuned"

# Get the latest version of the MLflow model
model_version = 4

# Name of the registered MLflow model
model_name = f"{catalog}.{schema}.{model_name}"

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

optimizable_info = requests.get(
  url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}",
  headers=headers).json()
print(optimizable_info)
if 'optimizable' not in optimizable_info or not optimizable_info['optimizable']:
   raise ValueError("Model is not eligible for provisioned throughput")

# COMMAND ----------


chunk_size = optimizable_info['throughput_chunk_size']

# Minimum desired provisioned throughput
min_provisioned_throughput = 1 * chunk_size

# Maximum desired provisioned throughput
max_provisioned_throughput = 2 * chunk_size

# Send the POST request to create the serving endpoint
data = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version,
                "min_provisioned_throughput": min_provisioned_throughput,
                "max_provisioned_throughput": max_provisioned_throughput,
            }
        ]
    },
}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers
)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.
