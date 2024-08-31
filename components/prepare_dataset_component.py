import kfp.dsl as dsl
from kfp.v2.dsl import component, Output, Dataset

@component(
    base_image='python:3.8',
    packages_to_install=['datasets']
)
def prepare_dataset(output_dataset: Output[Dataset]):
    from datasets import load_dataset

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    # Save the dataset to the specified output path
    dataset.to_csv(output_dataset.path + ".csv")
