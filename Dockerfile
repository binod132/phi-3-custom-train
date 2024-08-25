# Start with a GPU base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:12.1.0-base-ubi8
#FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubi8
#FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
#FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
#FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310

# Install any additional dependencies
RUN pip install --upgrade pip && \
    pip install transformers datasets torch google-cloud-storage && \
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
    pip install --no-deps xformers trl peft accelerate bitsandbytes datasets && \
    pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy the training script
COPY train.py /app/train.py

# Set the working directory
WORKDIR /app

# Command to run the training script
CMD ["python3", "train.py"]
