#!/bin/bash

BASE_DIR=`pwd`
SOURCE_DIR="$BASE_DIR/src"

# Set source corpus
SOURCE_GENERATOR_FILE="`dirname $0`/src/generate_source.py"
python $SOURCE_GENERATOR_FILE

# Run
python "$BASE_DIR/run.py"
