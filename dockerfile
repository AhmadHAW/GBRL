# adjusted from https://github.com/ToluClassics/transformers_notebook/blob/main/Dockerfile
FROM jupyter/base-notebook

ENV HF_HOME=/home/jovyan/hf_home

# Add RUN statements to install packages as the $NB_USER defined in the base images.

# Add a "USER root" statement followed by RUN statements to install system packages using apt-get,
# change file permissions, etc.

# If you do switch to root, always be sure to add a "USER $NB_USER" command at the end of the
# file to ensure the image runs as a unprivileged user by default.

USER root

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    jupyter \
    tensorflow-cpu \
    torch \ 
    torchvision \
    torchaudio \
    jax \
    jaxlib \
    optax

RUN python3 -m pip install --no-cache-dir \
    transformers \
    datasets\
    nltk \
    pytorch_lightning \
    gradio \
    sentencepiece \
    seqeval


USER ${NB_UID}