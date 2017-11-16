# DuReader Dataset
DuReader is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:
 - Real question
 - Real article
 - Real answer
 - Real application scenario
 - Rich annotation

# DuReader Baseline Systems
DuReader system implements 2 classic reading comprehension models([BiDAF](https://arxiv.org/abs/1611.01603) and [Match-LSTM](https://arxiv.org/abs/1608.07905)) on [DuReader dataset](https://ai.baidu.com//broad/subordinate?dataset=dureader). The system is implemented with 2 frameworks: [PaddlePaddle](http://paddlepaddle.org) and [TensorFlow](https://www.tensorflow.org).

## How to Run
### Download the Dataset
To Download DuReader dataset:
```
cd data && bash download.sh
```
For more details about DuReader dataset please refer to [DuReader Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader).

### Download Thirdparty Dependencies
We use Bleu and Rouge as evaluation metrics, the calculation of these metrics relies on the scoring scripts under "https://github.com/tylin/coco-caption", to download them, run:

```
cd utils && bash download_thirdparty.sh
```

### Preprocess the Data
After the dataset is downloaded, there is still some work to do to run the baseline systems. DuReader dataset offers rich amount of documents for every user question, the documents are too long for popular RC models to cope with. In our baseline models, we preprocess the train set and development set data by selecting the paragraph that is most related to the answer string, while for inferring(no available golden answer), we select the paragraph that is most related to the question string. The preprocessing strategy is implemented in `utils/preprocess.py`. To preprocess the raw data, you should first segment 'question', 'title', 'paragraphs' and then store the segemented result into 'segmented_question', 'segmented_title', 'segmented_paragraphs' like the downloaded preprocessed data, then run:
```
cat data/raw/trainset/search.train.json | python utils/preprocess.py > data/preprocessed/trainset/search.train.json
```
The preprocessed data can be automatically downloaded by `data/download.sh`, and is stored in `data/preprocessed`, the raw data before preprocessing is under `data/raw`.

### Run PaddlePaddle

#### Get the Vocab File

Once the preprocessed data is ready, you can run `utils/get_vocab.py` to generate the vocabulary file, for example, if you want to train model with Baidu Search data:
```
python utils/get_vocab.py --files data/preprocessed/trainset/search.train.json data/preprocessed/devset/search.dev.json  --vocab data/vocab.search
```

If you want to use the demo data, run:
```
python utils/get_vocab.py --files data/demo/trainset/search.train.json data/demo/devset/search.dev.json  --vocab data/demo/vocab.search
```

#### Environment Requirements
For now we've only tested on PaddlePaddle v0.10.5, to install PaddlePaddle and for more details about PaddlePaddle, see [PaddlePaddle Homepage](http://paddlepaddle.org).

#### Training
We implement 3 models with PaddlePaddle: Match-LSTM, BiDAF, and a classification model for data with `query_type='YES_NO'`, the model simply replaces the Pointer-Net on top of Match-LSTM model with a one-layered classifier. The 3 implemented models can all be trained and inferred by run `run.py`, to specify the model to train or to infer, use `--algo [mlstm|bidaf|yesno]`, for complete usage run `python run.py -h`.

The basic training and inference process has been wrapped in `run.sh`,  the basic usage is:
```
bash run.sh EXPERIMENT_NAME ALGO_NAME TASK_NAME
```
`EXPERIMENT_NAME` can be any legal folder name,  `ALGO_NAME` should be `bidaf`, `mlstm` or `yesno` for the 3 models have been implemented.
For example, to train a model with BiDAF, run:
```
bash run.sh test_bidaf bidaf train
```
`run.sh` creates a folder named `models`, and for every experiment a folder named `EXPERIMENT_NAME` is created under models, the basic experiment folder layout should be like:
```
models
└── test_bidaf
    ├── env
    ├── infer
    ├── log
    └── models
```
For training, all scripts the experiment uses will first be copied to `env`, and then run from there, and inference process is also run from `env`. `infer` folder keeps the result file created by inference, `log` folder keeps training and inference logs, and `models` folder keeps the models saved during training.

*Because our datatset and the model capacity is very large, if it's out of your device's capacity to successfully run the whole process, you can try with the shipped demo data, just use `run_demo.sh` for training and inferring,the usage is the same as `run.sh`*

#### Inference
To infer a trained model, run the same command as training and change `train` to `infer`,  and add `--testset <path_to_testset>` argument. for example, suppose the 'test_bidaf' experiment is successfully trained,  to infer the saved models, run:
```
bash run.sh test_bidaf bidaf infer --testset ../data/preprocessed/testset/search.test.json
```
The results corresponding to each model saved is under `infer` folder, and the evaluation metrics is logged into the infer log files under `log`.

Note if you want to infer a 'yesno' model, please sepecify an inferred result of a RC model, i.e. 'bidaf' or 'mlstm', under `models/SOME_RC_MODEL/infer/`, to `--testset`, because the 'yesno' model need the result answer of a RC model as its input.

#### Test result submission
You can infer and evaluate your models on development data set locally by following the above steps, once you've developed a model that works to your expectation on the dev set, we highly recommend you to submit your inference result on the released test set to us to evaluate. To get inference file on test set:

1. make sure the training is over.
2. infer your models on dev set and pick the best model.
3. keep only the best model under `models/<EXPERIMENT_NAME>/models`.
4. infer again with test set.
5. [submit the infer result file](http://ai.baidu.com/broad/submission?dataset=dureader).

### Run Tensorflow

We also implements the BIDAF and Match-LSTM models based on Tensorflow 1.0. You can refer to the [official guide](https://www.tensorflow.org/versions/r1.0/install/) for the installation of Tensorflow. The complete options for running our Tensorflow program can be accessed by using `python run.py -h`. Here we demonstrate a typical workflow as follows: 

#### Preparation
Before training the model, we have to make sure that the data is ready. For preparation, we will check the data files, make directories and extract a vocabulary for later use. You can run the following command to do this with a specified task name:

```
python run.py --prepare
```
You can specify the files for train/dev/test by setting the `train_files`/`dev_files`/`test_files`. By default, we use the data in `data/demo/`

#### Training
To train the reading comprehension model, you can specify the model type by using `--algo [BIDAF|MLSTM]` and you can also set the hyper-parameters such as the learning rate by using `--learning_rate NUM`. For example, to train a BIDAF model for 10 epochs, you can run:

```
python run.py --train --algo BIDAF --epochs 10
```

The training process includes an evaluation on the dev set after each training epoch. By default, the model with the least Bleu-4 score on the dev set will be saved.

#### Evaluation
To conduct a single evaluation on the dev set with the the model already trained, you can run the following command:

```
python run.py --evaluate --algo BIDAF
```

#### Prediction
You can also predict answers for the samples in some files using the following command:

```
python run.py --predict --algo BIDAF --test_files ../data/demo/search.dev.json
```

By default, the results are saved at `../data/results/` folder. You can change this by specifying `--result_dir DIR_PATH`.

## Copyright and License
Copyright 2017 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
