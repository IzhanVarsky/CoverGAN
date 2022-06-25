FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip cmake
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y cmake

RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY ./requirements.txt ./requirements.txt
COPY ./diffvg ./diffvg
WORKDIR .

RUN pip3 install -r ./requirements.txt
RUN cd diffvg && python3 ./setup.py install && cd ..
# If problems with installing or using diffvg, check this issue:
# https://github.com/BachiLi/diffvg/issues/29#issuecomment-994807865

# Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# extension builds significantly faster as we only compile for the
# currently active GPU configuration.
#RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
#ENTRYPOINT ["/entry.sh"]

ENTRYPOINT ["./covergan_training_command.sh"]