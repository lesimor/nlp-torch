#!/usr/bin/env bash

BASE_DIR=`pwd`
SOURCE_DIR="$BASE_DIR/src"

ZIP_FILE=0.2.0.zip
if test -f "$ZIP_FILE"; then
    echo "$ZIP_FILE exist"
else
    wget https://github.com/facebookresearch/fastText/archive/$ZIP_FILE
fi

UNZIP_FILE=fastText-0.2.0
if test -d "$UNZIP_FILE"; then
    echo "$UNZIP_FILE exist"
else
    unzip $ZIP_FILE
    cd $UNZIP_FILE
    make
fi

# Set source corpus
SOURCE_GENERATOR_FILE="`dirname $0`/generate_source.py"
python $SOURCE_GENERATOR_FILE

cd "$BASE_DIR/$UNZIP_FILE"
./fasttext supervised -input "$SOURCE_DIR/test.txt" -output "$BASE_DIR/model" -dim 2
