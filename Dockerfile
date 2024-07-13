FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    tmux &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists
  
