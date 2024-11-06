# Use a lightweight base image with CUDA support for PyTorch
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install dependencies in one command to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install common dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install runpod==1.7.4

# Copy only requirements files for caching before copying the entire directory
COPY ComfyUI/requirements.txt /ComfyUI/requirements.txt
COPY ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt
COPY ComfyUI/custom_nodes/comfyui_densepose/requirements.txt /ComfyUI/custom_nodes/comfyui_densepose/requirements.txt
COPY ComfyUI/custom_nodes/comfyui_face_parsing/requirements.txt /ComfyUI/custom_nodes/comfyui_face_parsing/requirements.txt
# COPY ComfyUI/custom_nodes/ComfyUI-Impact-Pack/requirements.txt /ComfyUI/custom_nodes/ComfyUI-Impact-Pack/requirements.txt
COPY ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt /ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt
COPY ComfyUI/custom_nodes/ComfyUI-tbox/requirements.txt /ComfyUI/custom_nodes/ComfyUI-tbox/requirements.txt
COPY ComfyUI/custom_nodes/rgthree-comfy/requirements.txt /ComfyUI/custom_nodes/rgthree-comfy/requirements.txt

# Install all dependencies in a single layer
RUN python -m pip install -r /ComfyUI/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/comfyui_face_parsing/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/ComfyUI-Impact-Pack/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/ComfyUI-tbox/requirements.txt --no-cache-dir
RUN python -m pip install -r /ComfyUI/custom_nodes/rgthree-comfy/requirements.txt --no-cache-dir

# Copy the full application code as the last step
# COPY ComfyUI /ComfyUI
# WORKDIR /ComfyUI
WORKDIR /runpod-volume/ComfyUI

# Define the command to run the application
CMD ["python", "-u", "handler.py"]
