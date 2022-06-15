FROM python:3.7

EXPOSE 5001

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip cmake
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y cmake

# Install PyTorch
RUN pip3 install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0

COPY ./requirements.txt ./requirements_covergan.txt
COPY ./covergan/requirements.txt ./requirements_server.txt
WORKDIR .
# Install other Python libraries
RUN pip install -r ./requirements_covergan.txt
RUN pip install -r ./requirements_server.txt

# Clone and build DiffVG
# If problems with installing or using diffvg, check this issue:
# https://github.com/BachiLi/diffvg/issues/29#issuecomment-994807865
RUN git clone --recursive https://github.com/BachiLi/diffvg
RUN cd diffvg && python setup.py install

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libmagickwand-dev

# Install fonts
COPY ./covergan/fonts /usr/share/fonts
RUN fc-cache -f -v

ENTRYPOINT ["./entry.sh"]
