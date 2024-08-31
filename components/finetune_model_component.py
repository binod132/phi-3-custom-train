import kfp.dsl as dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component(
    base_image='us-docker.pkg.dev/brave-smile-424210-m0/train/phi-3:dev'
)
def finetune_model(dataset: Input[Dataset], output_model: Output[Model]):
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import is_bfloat16_supported
    from datasets import load_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct",
        dtype=None,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    #model.to(device)

    # Load and prepare dataset
    dataset = load_dataset('csv', data_files=dataset.path)
    
    # Define training arguments
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=20,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=output_model.path,
        ),
    )

    # Train and save the model
    trainer.train()
    trainer.save_model(output_model.path)
