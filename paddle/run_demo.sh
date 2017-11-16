#!/bin/bash
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -x
set -e

PWD=$(pwd)
ROOT=$(dirname $PWD)

model=$1; shift;
algo=$1; shift;
job=$1; shift;

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


emb_dim=100
vocab_size=10000

train() {
    cp *.py $env_dir/
    PYTHONPATH=$PWD:$ROOT CUDA_VISIBLE_DEVICES=0 python $env_dir/run.py \
        --trainset ../data/demo/trainset/search.train.json \
        --testset ../data/demo/devset/search.dev.json \
        --vocab_file ../data/demo/vocab.search \
        --emb_dim $emb_dim \
        --batch_size 32 \
        --vocab_size $vocab_size \
        --trainer_count 1 \
        --log_period 10 \
        --test_period 100 \
        --num_passes 2 \
        --use_gpu \
        --save_dir $model_dir \
        --algo $algo \
        $@ \
        --saving_period 1000 2>&1 | tee $log_dir/train.log
}

infer() {
    model_name=`basename $2`
    PYTHONPATH=$PWD:$ROOT CUDA_VISIBLE_DEVICES=0 python $env_dir/run.py \
        --vocab_file ../data/demo/vocab.search \
        --emb_dim $emb_dim \
        --batch_size 32 \
        --vocab_size $vocab_size \
        --trainer_count 1 \
        --use_gpu \
        --is_infer \
        $@ \
        --algo $algo 2>&1 | tee $log_dir/infer.$model_name.log
}

dir_infer() {
    dir=$1; shift;
    for f in $( ls -t $dir );
    do
        model_file=$dir/$f
        infer --model_file $model_file $@
    done
}


if [ $job == "train" ]; then
    train $@
else
    dir_infer $model_dir $@
fi
