FROM rust:1.52.1 as builder

# Installing `rustup`:
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build ProtoSVG
WORKDIR /usr/src/protosvg
COPY ./protosvg .
RUN rustup component add rustfmt
RUN cargo install --locked --path .

FROM python:3.7

EXPOSE 5001
EXPOSE 50051

# System dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libmagic1 \
        libraqm-dev \
        supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the built ProtoSVG
COPY --from=builder /usr/local/cargo/bin/protosvg /usr/bin/protosvg

# Install PyTorch
#RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#RUN pip3 install torch torchvision torchaudio

# Install other Python libraries
COPY ./requirements.txt /inference-api/requirements.txt
RUN pip install -r /inference-api/requirements.txt

# Clone and build DiffVG
# If problems with installing or using diffvg, check this issue:
# https://github.com/BachiLi/diffvg/issues/29#issuecomment-994807865
WORKDIR /tmp/builds
RUN git clone --recursive https://github.com/BachiLi/diffvg
RUN cd diffvg && python setup.py install
RUN rm -rf /tmp/builds

# Configure the supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy CoverGAN
#COPY ./covergan /inference-api/covergan
COPY ./covergan/captions /inference-api/covergan/captions
COPY ./covergan/colorer /inference-api/covergan/colorer
COPY ./covergan/fonts /inference-api/covergan/fonts
COPY ./covergan/outer /inference-api/covergan/outer
COPY ./covergan/protosvg /inference-api/covergan/protosvg
COPY ./covergan/scripts /inference-api/covergan/scripts
COPY ./covergan/utils /inference-api/covergan/utils
COPY ./covergan/*.py /inference-api/covergan/
COPY ./covergan/weights /inference-api/covergan/weights

WORKDIR /inference-api/covergan

# Copy backend files
#COPY . .
COPY ./gen.py ./gen.py
COPY ./server.py ./server.py
COPY ./config.yml ./config.yml
COPY ./fonts_cfg.py ./fonts_cfg.py
COPY ./service_utils.py ./service_utils.py

# Run the processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
