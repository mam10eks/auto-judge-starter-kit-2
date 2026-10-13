FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/

WORKDIR /auto-judge

# Install into the base image's /venv (its PATH runs /venv/bin) — a --system
# install would be invisible at runtime, breaking any dependency added here.
RUN pip install -e .[all] --break-system-packages

RUN python3 -c "import nltk; nltk.download('stopwords')"

ENV HF_HUB_OFFLINE=1

# git metadata for provenance (tira's runtime stats look for a repo at ./)
ADD .git /auto-judge/.git

