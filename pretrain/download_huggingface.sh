#!/usr/bin/env bash

NAME=$1
SAVEDIR=$2

export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download \
--resume-download $NAME \
--local-dir $SAVEDIR \
--exclude *.safetensors *.h5 \
${@:3}
