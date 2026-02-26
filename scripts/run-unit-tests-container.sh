#!/bin/bash

set -e

folder=/test/unit

cd /build && \
mkdir -p $folder && \

model_dir="/build/models/phi3.5"

./scripts/download-phi.sh "$model_dir"

echo "Running tests..."

gotestsum --format testname --junitfile=$folder/unit.xml --jsonfile=$folder/unit.json -- -coverprofile=$folder/cover.out -coverpkg ./... -timeout 60m -race

echo Done.