FROM python:3.8

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # pipのアップデート
    && pip install --upgrade pip


RUN pip install --ignore-installed \
    # 上3つはendpointを立てる時に必要
    flask \
    gevent \
    gunicorn \
    # モデルに必要なモジュール
    pandas \
    numpy \
    scipy \
    sklearn \
    torch \
    pytorch-tabnet \
    catboost \
    joblib \
    lightgbm \
    xgboost \
    jupyter \
    jupyter_contrib_nbextensions \
    umap-learn \
    awscli \
    dataclasses \
    fsspec \
    sagemaker


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY model /opt/program
WORKDIR /opt/program

