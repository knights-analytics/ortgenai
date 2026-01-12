#!/bin/bash

set -e

model_dir="$1"

if [ -z "$model_dir" ]; then
    echo "Usage: $0 <model_directory>"
    exit 1
fi

model_url_base="https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
model_files=(
	"genai_config.json"
	"phi-3.5-v-instruct-embedding.onnx"
	"phi-3.5-v-instruct-embedding.onnx.data"
	"phi-3.5-v-instruct-vision.onnx"
    "phi-3.5-v-instruct-vision.onnx.data"
	"phi-3.5-v-instruct-text.onnx"
	"phi-3.5-v-instruct-text.onnx.data"
	"special_tokens_map.json"
	"tokenizer.json"
	"tokenizer_config.json"
    "processor_config.json"
)

mkdir -p "$model_dir"

echo "Downloading Phi-3.5-vision-instruct-onnx model if not present..."
for f in "${model_files[@]}"; do
	if [ ! -f "$model_dir/$f" ]; then
		echo "Downloading $model_url_base/$f..."
		curl -L -o "$model_dir/$f" "$model_url_base/$f"
	fi
done