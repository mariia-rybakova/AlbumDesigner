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


CMD ["python3", "-W", "ignore::UserWarning", "main.py"]