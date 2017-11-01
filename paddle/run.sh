#!/bin/bash

set -x
set -e

model=$1; shift;
algo=$1; shift;
job=$1; shift;

echo "rest args:$@"

model_root='./models'
model=$model_root/$model

mkdir -p $model_root
mkdir -p $model

save_dir=`readlink -f $model`

env_dir=$save_dir/env
model_dir=$save_dir/models
infer_dir=$save_dir/infer
log_dir=$save_dir/log

mkdir -p $env_dir
mkdir -p $model_dir
mkdir -p $infer_dir
mkdir -p $log_dir

train() {
    cp *.py $env_dir/
    cp *.sh $env_dir/
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $env_dir/run.py \
        --trainset ../data/baidu.v2/search.train.json \
        --testset ../data/baidu.v2/search.dev.min.json \
        --vocab_file ../data/baidu.v2/vocab.search \
        --emb_dim 300 \
        --batch_size 32 \
        --vocab_size 218967 \
        --trainer_count 2 \
        --log_period 10 \
        --test_period 100 \
        --use_gpu \
        --save_dir $model_dir \
        --algo $algo \
        $@ \
        --saving_period 1000 2>&1 | tee $log_dir/train.log
}

infer() {
    model_name=`basename $2`
    CUDA_VISIBLE_DEVICES=3 python $env_dir/run.py \
        --trainset ../data/baidu.v2/search.train.json \
        --testset ../data/baidu.v2/search.dev.json \
        --vocab_file ../data/baidu.v2/vocab.search \
        --emb_dim 300 \
        --batch_size 32 \
        --vocab_size 218967 \
        --trainer_count 1 \
        --use_gpu \
        --metric 'marco' \
        --is_infer \
        $@ \
        --algo $algo 2>&1 | tee $log_dir/infer.$model_name.log
}

dir_infer() {
    dir=$1; shift;
    for f in $( ls -t $dir );
    do
        model_file=$dir/$f
        infer --model_file $model_file
    done
}

echo "rest args: $@"

if [ $job == "train" ]; then
    train $@
else
    dir_infer $model_dir
fi
