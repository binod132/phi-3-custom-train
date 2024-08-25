# Start with a GPU base image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Install any additional dependencies
RUN pip install --upgrade pip && \
    pip install transformers datasets torch google-cloud-storage && \
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
    pip install --no-deps xformers trl peft accelerate bitsandbytes datasets
#pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy the training script
COPY train.py /app/train.py

# Set the working directory
WORKDIR /app

# Command to run the training script
CMD ["python3", "train.py"]
