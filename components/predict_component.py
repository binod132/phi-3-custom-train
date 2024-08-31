import kfp.dsl as dsl
from kfp.v2.dsl import component, Input, Model

@component(
    base_image='pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel'
)
def predict(model: Input[Model], text: str) -> str:
    import torch
    from unsloth import FastLanguageModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = FastLanguageModel.from_pretrained(model.path)
    #model.to(device)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    return tokenizer.decode(outputs[0])
