import kfp.dsl as dsl
from kfp.v2.dsl import component, Output, Dataset

@component(
    base_image='us-docker.pkg.dev/brave-smile-424210-m0/train/phi-3:dev',
    packages_to_install=['datasets']
)
def prepare_dataset(output_dataset: Output[Dataset]):
    
    import os
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported
    
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # Model and tokenizer setup
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct",
        dtype = dtype,
        max_seq_length=2048,
        load_in_4bit=True,
    )
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
