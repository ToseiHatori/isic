FROM gcr.io/kaggle-gpu-images/python:latest

# Set the environment variable for CUDA
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --allow-change-held-packages \
    libnccl2=2.18.3-1+cuda12.1 \
    libnccl-dev=2.18.3-1+cuda12.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip && \
    pip install black jupyter-contrib-nbextensions && \
    jupyter contrib nbextension install && \
    jupyter nbextensions_configurator enable && \
    jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip && \
    jupyter nbextension enable jupyter-black-master/jupyter-blacks

# Verify installation
RUN nvcc --version && \
    python -c "import torch; print(torch.cuda.is_available())" && \
    dpkg -l | grep nccl
