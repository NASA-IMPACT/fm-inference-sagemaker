FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set working directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN add-apt-repository -y ppa:ubuntugis/ppa

RUN apt-get update && \
    apt-get install -y gcc g++ wget && \
    apt-get clean git && \
    apt-get install -y gdal-bin libgdal-dev && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1 && \
    apt-get install -y python3.12 python3-pip python3.12-venv python3.12-dev && \
    apt-get install -y cmake ninja-build sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.12 2
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1
RUN update-alternatives --set python /usr/bin/python3.12

RUN cd /tmp && \
    wget https://download.osgeo.org/proj/proj-9.3.0.tar.gz && \
    tar -xzf proj-9.3.0.tar.gz && \
    cd proj-9.3.0 && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_TIFF=OFF \
        -DENABLE_CURL=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_APPS=OFF && \
    make -j$(nproc) VERBOSE=1 && \
    make install && \
    cd /tmp && \
    rm -rf proj-9.3.0*

# Install GDAL from source (minimal build)
RUN wget http://download.osgeo.org/gdal/CURRENT/gdal-3.11.3.tar.gz
RUN tar zxvf gdal-3.11.3.tar.gz
RUN mkdir -p gdal-3.11.3/build && \
    cd gdal-3.11.3/build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON && \
    cmake --build ./ && \
    cmake --install ./ && \
    cd ../.. && rm -rf gdal-3.12*

# Set environment variables
ENV GDAL_DATA=/usr/local/share/gdal
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH

RUN python3.12 -m ensurepip --upgrade

# Upgrade pip and install uv
RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install uv

# Copy requirements and install Python dependencies
COPY floods/requirements.txt /app/

RUN uv pip install -r /app/requirements.txt --system

RUN mkdir /app/lib
# Copy the rest of the application
COPY shared/lib /app/lib

COPY shared/predictor.py /app/

# Expose port
EXPOSE 8001

# Default command for development (can be overridden by docker-compose)
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001"]
