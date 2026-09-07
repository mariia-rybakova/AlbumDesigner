FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get clean \
    && apt-get -y update \
    && apt-get -y upgrade

RUN apt-get -y install \
    python3-dev \
    nginx \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    vim \
    zip \
    python3-pip \
    python3-venv \
    unzip


ENV PYTHONWARNINGS="ignore"
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH

ENV PATH="$VENV_PATH/bin:$PATH"

#Set working directory to app
WORKDIR /usr/app

#Copy over files to app directory
COPY ./requirements.txt ./

#Install requirements.txt
RUN $VENV_PATH/bin/pip install --upgrade pip
RUN $VENV_PATH/bin/pip install --no-cache-dir -r requirements.txt
#RUN pip install --force-reinstall git+https://github.com/pic-time/python-infra.git

#Copy rest of files over to working directory
COPY ./setup.py /usr/app/setup.py
COPY ./ptinfra /usr/app/ptinfra
RUN $VENV_PATH/bin/pip install ./

COPY ./ /usr/app/

# -----------------------------------------------------------------------------
# Narrator model: the trained albumNarrator policy (`select.narrator`), published
# as a zip bundle whose flat members are `policy.pt` and `attribute_axes.npz` --
# the two filenames CONFIGS['narrator'] points at once unzipped into
# files/narrator/. Built and cross-checked by tools/build_narrator_bundle.py
# (which verifies the checkpoint's clip_dim is 768 and that all four attribute
# axes are present, both of which fail silently otherwise).
#
# The weights are NOT in git: 46 MB of binary that changes on every retrain, so
# files/narrator/*.pt and *.npz are gitignored and delivered here instead.
#
# ADD never auto-extracts a *remote* archive, so unzip explicitly -- landing the
# zip itself at files/narrator/policy.pt would make torch.load fail with "file in
# archive is not in a subdirectory", the same trap bib-detection documents.
#
# The URL is versioned on purpose: a rebuild must not silently pick up different
# weights than the ones a run was validated against.
# -----------------------------------------------------------------------------
ADD 'https://a1devops1versions.blob.core.windows.net/ai-models/album-narrator/narrator_models_v1.zip?se=2123-08-31T07%3A37%3A22Z&sp=r&sv=2022-11-02&sr=b&sig=eNmidzNDDd3FS6XiOTUKYEwpBPGqiZxOSei5NzV5kOU%3D' ./files/narrator/models.zip
RUN unzip ./files/narrator/models.zip -d ./files/narrator && rm ./files/narrator/models.zip


CMD ["python3", "-W", "ignore::UserWarning", "main.py"]