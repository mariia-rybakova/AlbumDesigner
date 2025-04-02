FROM ubuntu:22.04
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
    unzip
#Set working directory to app
WORKDIR /usr/app

#Copy over files to app directory
COPY ./requirements.txt ./

#Install requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --force-reinstall git+https://github.com/pic-time/python-infra.git

#Copy rest of files over to working directory
COPY ./setup.py /usr/app/setup.py
COPY ./ptinfra /usr/app/ptinfra
RUN pip install ./

COPY ./ /usr/app/


CMD ["python3", "-W", "ignore::UserWarning", "main.py"]