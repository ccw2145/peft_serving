# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune `meta_llama_3_8b` model from marketplace with QLORA
# MAGIC
# MAGIC Databricks hosts the `meta_llama_3_8b` in [Databricks marketplace](https://marketplace.databricks.com/). This is a tutorial to show how to fine tune the models on the [mosaicml/dolly_hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 16.1 GPU ML Runtime
# MAGIC - Instance: `g5.8xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure, `g2-standard-8` or `a2-highgpu-1g` on GCP
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient fine tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# MAGIC %pip install bitsandbytes==0.45
# MAGIC %pip install accelerate
# MAGIC # !pip install git+https://github.com/huggingface/peft.git
# MAGIC %pip install peft
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# !nvidia-smi

# COMMAND ----------

# ! export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] ='max_split_size_mb:128'
dbutils.widgets.text("huggingface_token", "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", "Enter Parameter")

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import mlflow

# torch.cuda.empty_cache()


# COMMAND ----------

# torch.cuda.memory_summary() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the model
# MAGIC
# MAGIC In this section we will load the [meta_llama_3_8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model installed from [Databricks marketplace](https://marketplace.databricks.com/) saved in Unity Catalog to local disk, quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

os.environ['HF_TOKEN'] = dbutils.widgets.get("huggingface_token")

# COMMAND ----------

base_model_path = '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/meta_llama_3_8b_base_16b'
base_tokenizer_path = '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/meta_llama_3_tokenzier'
adapters_path = '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/adapters'
peft_model_path = 'meta_llama_3_8b_lora_fine_tune'
peft_model_path_spanish = 'valadapt-meta-llama-3.1-8b-spanish'

adapters_mapping_path = '/Volumes/cindy_demo_catalog/llm_fine_tuning/trained_models/adapters_mapping.json'


# COMMAND ----------

# import json

# adapters_mapping = {
#     "Dolly": peft_model_path,
#     "Spanish": peft_model_path_spanish
# }

# with open(adapters_mapping_path, 'w') as f:
#     json.dump(adapters_mapping, f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Model Locally

# COMMAND ----------

# from peft import PeftModel

# base_model_id = 'meta-llama/Llama-3.1-8B'
# base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cuda:0",torch_dtype=torch.float16)
# base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda:0",torch_dtype=torch.float16)
# model = PeftModel.from_pretrained(base_model, "faridlazuarda/valadapt-meta-llama-3.1-8b-spanish")
# tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
# model = PeftModel.from_pretrained(base_model, "faridlazuarda/valadapt-meta-llama-3.1-8b-spanish")

# COMMAND ----------

# MAGIC %md
# MAGIC ### PyFunc Flavor

# COMMAND ----------

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, PeftConfig

class FINETUNED_QLORA(mlflow.pyfunc.PythonModel):
    # Load base model, tokenizer, and adapters.
    def load_context(self, context):
        import json
        import os
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        )
        # Load the tokenizer and set the pad token to the EOS token.
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['tokenizer'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the base model (using 4-bit quantization in this example).
        self.base_model = AutoModelForCausalLM.from_pretrained(
            context.artifacts['base_model'], 
            return_dict=True, 
            quantization_config=bnb_config, 
            device_map={"": 0},
            # trust_remote_code=True,
        )
        # Load PEFT adapters from a dictionary artifact.
        # The "adapters" artifact should be a dict mapping adapter names to adapter paths.
        with open(context.artifacts["adapters_mapping"], "r") as f:
            self.adapters_mapping = json.load(f)

        print('loaded adapter mappings')
        # self.adapters_mapping["Dolly"]
        # self.path2_to_check = f"{context.artifacts["adapters"]}/{self.adapters_mapping["Dolly"]}"
    # def predict(self, context, model_input):
    #     import os
    #     return [self.path_to_check, os.path.exists(self.path_to_check), self.path2_to_check, os.path.exists(self.path2_to_check)]

        # Use the base model as the starting point.
        # self.model = self.base_model
        self.model = PeftModel.from_pretrained(self.base_model,f"{context.artifacts["adapters"]}/{self.adapters_mapping["Dolly"]}", adapter_name="default", local_files_only=True)

        for adapter_name, adapter_path in self.adapters_mapping.items():
            # This loads the adapter onto the model under the provided adapter name.
            self.model.load_adapter(f"{context.artifacts["adapters"]}/{adapter_path}", adapter_name=adapter_name)
            print('loaded Peft Model',f"{context.artifacts["adapters"]}/{adapter_path}")

        # Set the model to evaluation mode.
        self.model.eval()
        print(self.model)

    def predict(self, context, model_input):
        # Extract input values.
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]
        adapter_name = model_input.get("adapter", ['None'])[0]
        print('input:', prompt, temperature, max_tokens, adapter_name)
        # print(list(self.adapters_mapping.keys()))
        # Activate the desired adapter if provided.
        if adapter_name in list(self.adapters_mapping.keys()):
          self.model.set_adapter(adapter_name)
          print('using adapter:', self.model.get_model_status())

        else:
          print('no adapter')
          generated_text = ''
          return generated_text
        
        # Tokenize the input prompt with padding and truncation, and move to CUDA.
        batch = self.tokenizer(prompt, padding=True, truncation=True, return_tensors='pt').to('cuda')
        
        # Use mixed precision inference on CUDA.
        with torch.amp.autocast('cuda'):
            output_tokens = self.model.generate(
                input_ids=batch.input_ids, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the generated tokens into text.
        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        return generated_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local testing

# COMMAND ----------

# Create a local model context object with required artifact paths for testing
# import json
# adapters_mapping = {
#         'Dolly': peft_model_path,
#         'Spanish': peft_model_path_spanish,
#                 }
artifacts = {
    "tokenizer": base_tokenizer_path,      
    "base_model": base_model_path,  
    "adapters_mapping": adapters_mapping_path,
    "adapters": adapters_path
            }

class ModelContext:
    def __init__(self):
        self.artifacts = artifacts
# Instantiate a dummy context.
dummy_context = ModelContext()

# COMMAND ----------

# Instantiate your pyfunc wrapper and load the context.
finetuned_model = FINETUNED_QLORA()
finetuned_model.load_context(dummy_context)

# COMMAND ----------

test_input = {
    "prompt": ["What is ML?"],
    "temperature": [0.1],
    "max_tokens": [100],
    "adapter": ["default"]
}

# Run a prediction and display the output.
print("Testing prediction...")
generated_output = finetuned_model.predict(dummy_context, test_input)
print("Generated Output:", generated_output)

# COMMAND ----------



# COMMAND ----------

test_input = {
    "prompt": ["what is Databricks?"],
    "temperature": [0.1],
    "max_tokens": [100],
    "adapter": ["Spanish"]
}

# Run a prediction and display the output.
print("Testing prediction...")
generated_output = finetuned_model.predict(dummy_context, test_input)
print("Generated Output:", generated_output)

# COMMAND ----------

test_input = {
    "prompt": ["what is Databricks?"],
    "temperature": [0.1],
    "max_tokens": [100],
    "adapter": ["Dolly"]
}

# Run a prediction and display the output.
print("Testing prediction...")
generated_output = finetuned_model.predict(dummy_context, test_input)
print("Generated Output:", generated_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log to MLFlow + UC

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature

# Set mlflow registry to databricks-uc
mlflow.set_registry_uri("databricks-uc")
# Specify an input example that conforms to the input schema for the task.
import numpy as np
input_example={"prompt": ["Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"],
               "temperature": [0.5], 
               "max_tokens": [100],
               'adapter': ['None']}
output_example = "Apache Spark is an open-source data processing engine that is designed to efficiently process large data sets."
# Define input and output schema
# Define input and output schema
input_schema = Schema([
    ColSpec("string", "prompt"),
    ColSpec("double", "temperature"),
    ColSpec("long",   "max_tokens"),
    ColSpec("string", "adapter")
])
# output_schema = Schema([ColSpec(DataType.string)])        
# signature = infer_signature(input_example, output_example)
output_schema = Schema([ColSpec("string", "generated_text")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

signature 

# COMMAND ----------

# catalog = "cindy_demo_catalog"
# schema = "llm_fine_tuning"
# model_name = "pyfun_logging_test"
# registered_model_name = f"{catalog}.{schema}.{model_name}"

# artifacts = {
#     "tokenizer": base_tokenizer_path,      
#     "base_model": base_model_path,  
#     "adapters_mapping": adapters_mapping_path,
#     "adapters": adapters_path
#             }

# with mlflow.start_run() as run:  
#     mlflow.pyfunc.log_model(
#         "model",
#         python_model=FINETUNED_QLORA(),
#         artifacts= artifacts,
#         pip_requirements=["torch==2.5.0", "torchvision==0.20.0","transformers==4.46.3", "accelerate==1.1.1", "peft==0.15.2", "bitsandbytes==0.45.0"],
#         input_example= input_example,
#         signature=signature,
#         registered_model_name=registered_model_name
#     )

# COMMAND ----------

catalog = "cindy_demo_catalog"
schema = "llm_fine_tuning"
model_name = "meta_llama_3_8b_lora_fine_tuned_pyfunc"
registered_model_name = f"{catalog}.{schema}.{model_name}"


artifacts = {
    "tokenizer": base_tokenizer_path,      
    "base_model": base_model_path,  
    "adapters_mapping": adapters_mapping_path,
    "adapters": adapters_path
            }


with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=FINETUNED_QLORA(),
        artifacts= artifacts,
        pip_requirements=["torch==2.5.0", "torchvision==0.20.0","transformers==4.46.3", "accelerate==1.1.1", "peft==0.15.2", "bitsandbytes==0.45.0"],
        input_example= input_example,
        signature=signature,
        registered_model_name=registered_model_name
    )

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/14")

# COMMAND ----------

pd.DataFrame(input_dict)

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np

# loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/6")


prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
what is Databricks?

### Response: """

input_dict = {
    "prompt": [prompt],
    "temperature": [0.5],
    "max_tokens": [50],
    "adapter": [""],
}
preds = loaded_model.predict(input_dict)

# COMMAND ----------

preds

# COMMAND ----------

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
What is ML?

### Response: """

test_input = {
    "prompt":prompt,
    "temperature": [0.5],
    "max_tokens": [100],
    "adapter": ['Doll']
}

# Load model as a PyFuncModel.

text_example=pd.DataFrame(test_input)

# Predict on a Pandas DataFrame.
response = loaded_model.predict(text_example)
print(response)

# COMMAND ----------

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
what is kangen water?

### Response: """

test_input = {
    "prompt":prompt,
    "temperature": [0.1],
    "max_tokens": [100],
    "adapter": ['']
}

# Load model as a PyFuncModel.

text_example=pd.DataFrame(test_input)

# Predict on a Pandas DataFrame.
response = loaded_model.predict(text_example)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve Registered Model

# COMMAND ----------

import requests
import json

# Set the name of the MLflow endpoint
endpoint_name = "cindy_llama3_8b_pyfunc_multi_adapters"

# Get the latest version of the MLflow model
model_version = 2

# Name of the registered MLflow model
registered_model_name = f"{catalog}.{schema}.{model_name}"

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

optimizable_info = requests.get(
  url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{registered_model_name}/{model_version}",
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
