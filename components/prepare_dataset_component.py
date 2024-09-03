import kfp.dsl as dsl
from kfp.v2.dsl import component, Output, Dataset
import os
import zipfile
import subprocess

@component(
    base_image='us-docker.pkg.dev/brave-smile-424210-m0/train/phi-3:dev',
    packages_to_install=["transformers", "google-cloud-storage"]
)
def prepare_dataset(output_dataset: Output[Dataset]):
    
    import os
    import subprocess
    import shutil
    from google.cloud import storage
    from transformers import LlamaTokenizerFast
    #import torch
    #from unsloth import FastLanguageModel
    from datasets import load_dataset
    import pandas as pd
    from google.cloud import storage
    #from transformers import TrainingArguments
    #from trl import SFTTrainer
    #from unsloth import is_bfloat16_supported
    
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # Model and tokenizer setup
    #model, tokenizer = FastLanguageModel.from_pretrained(
    #    model_name="unsloth/Phi-3-mini-4k-instruct",
    #    dtype = dtype,
    #    max_seq_length=2048,
    #    load_in_4bit=True,
    #)
    #load tokenizer
    def download_blob(bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    bucket_name = 'phi-3-custom-train'
    source_blob_name = 'tokenizerphi3s.zip'
    destination_file_name = '/tmp/tokenizerphi3s.zip'
    download_blob(bucket_name, source_blob_name, destination_file_name)

    tokenizer_dir="/tmp/tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    shutil.unpack_archive(destination_file_name, tokenizer_dir, 'zip')

    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)

    #data preprocessing prompt
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    from datasets import load_dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    dataset.to_csv(output_dataset.path + ".csv")
