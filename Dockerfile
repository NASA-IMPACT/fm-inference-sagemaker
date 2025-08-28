# Use the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Set working directory
WORKDIR /var/task

# Install system dependencies
RUN dnf install -y gcc && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get update && apt-get install -y libgl1 python3-pip python3-dev git libgdal-dev --fix-missing
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN pip3 install --upgrade pip

# RUN pip3 install GDAL

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --ignore-installed

ENV CUDA_HOME=/usr/local/cuda

RUN mkdir models

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
COPY code /opt/program

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
WORKDIR /opt/program

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
