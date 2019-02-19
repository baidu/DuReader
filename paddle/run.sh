#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

paragraph_extraction ()
{
    SOURCE_DIR=$1
    TARGET_DIR=$2
    echo "Start paragraph extraction"
    echo "Source dir: $SOURCE_DIR"
    echo "Target dir: $TARGET_DIR"
    mkdir -p $TARGET_DIR/trainset
    mkdir -p $TARGET_DIR/devset
    mkdir -p $TARGET_DIR/testset

    cat $SOURCE_DIR/trainset/search.train.json | python paragraph_extraction.py train \
            > $TARGET_DIR/trainset/search.train.json
    cat $SOURCE_DIR/trainset/zhidao.train.json | python paragraph_extraction.py train \
            > $TARGET_DIR/trainset/zhidao.train.json

    cat $SOURCE_DIR/devset/search.dev.json | python paragraph_extraction.py dev \
            > $TARGET_DIR/devset/search.dev.json
    cat $SOURCE_DIR/devset/zhidao.dev.json | python paragraph_extraction.py dev \
            > $TARGET_DIR/devset/zhidao.dev.json

    cat $SOURCE_DIR/testset/search.test.json | python paragraph_extraction.py test \
            > $TARGET_DIR/testset/search.testjson
    cat $SOURCE_DIR/testset/zhidao.test.json | python paragraph_extraction.py test \
            > $TARGET_DIR/testset/zhidao.testjson
    echo "Paragraph extraction done!"
}


PROCESS_NAME="$1"
case $PROCESS_NAME in
    --para_extraction)
    # Start paragraph extraction 
    if [ ! -d ../data/preprocessed ]; then
        echo "Please download the preprocessed data first (See README - Preprocess)"
        exit 1
    fi
    paragraph_extraction ../data/preprocessed ../data/extracted
    ;;
    --train|--evaluate|--predict)
        # Start Paddle baseline
        python run.py $@
    ;;
    *)
        echo $"Usage: $0 {--para_extraction|--train|--evaluate|--predict}"
esac
