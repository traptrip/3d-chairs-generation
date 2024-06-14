FROM python:3.10.14 AS python_stage

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_DIR=/app
WORKDIR "$APP_DIR"

ENV TZ=Europe/London \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

COPY --from=python_stage /usr/bin/python3 /usr/bin/python3

RUN apt-get update && apt-get install git -y && \
    git config --global http.sslverify false && \
    apt-get update && apt-get install gcc python3-pip -y && \
    apt-get install libgl1 libglib2.0-0 -y && \
    pip install --upgrade pip setuptools

RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117

COPY requirements.txt $APP_DIR/
RUN python3 -m pip install setuptools==69.5.1 
RUN python3 -m pip install --no-warn-script-location -r requirements.txt

COPY . $APP_DIR/

RUN python3 -m pip install --no-warn-script-location -r requirements-cuda.txt

CMD python3 gradio_app.py --unload-models --empty-cache
