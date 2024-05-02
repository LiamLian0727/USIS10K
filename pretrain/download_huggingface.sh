#!/usr/bin/env bash

NAME=$1

export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download \
--resume-download $NAME \
--exclude *.safetensors *.h5 \
${@:3}
