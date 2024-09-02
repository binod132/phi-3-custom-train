import kfp.dsl as dsl
from kfp.v2.dsl import pipeline
from kfp.v2 import compiler
from components.prepare_dataset_component import prepare_dataset
from components.finetune_model_component import finetune_model
from components.predict_component import predict

@pipeline(name="phi-3-model-finetuning-pipeline-new")
def pipeline():
    # Step 1: Prepare Dataset
    prepare_dataset_task = prepare_dataset()

    # Step 2: Fine-tune Model with GPU and Machine Type Configuration
    finetune_model_task = finetune_model(
        dataset=prepare_dataset_task.outputs["output_dataset"]
    ).set_cpu_limit('1')\
    .set_memory_limit('4')\
    .set_gpu_limit(1)\
    .set_accelerator_type('NVIDIA_TESLA_T4')

    # Step 3: Predict using the finetuned model
    #predict_task = predict(
    #    model=finetune_model_task.outputs["output_model"]
    #)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='phi_3_finetune_pipeline.json'
    )
