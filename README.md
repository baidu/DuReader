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

We implement a BiDAF model with PaddlePaddle. Note that we have an update on the PaddlePaddle baseline (Feb 25, 2019). The major updates have been noted in `paddle/UPDATES.md`. On the dataset of DuReader, the PaddlePaddle baseline has better performance than our Tensorflow baseline. 

The PaddlePaddle baseline includes the following procedures: paragraph extraction, vocabulary preparation, training, evaluation and inference. All these procedures have been wrapped in `paddle/run.sh`. You can start one procedure by running run.sh with specific arguments. The basic usage is:

```
sh run.sh --PROCESS_NAME --OTHER_ARGS
```

PROCESS\_NAME can be one of `para_extraction`, `prepare`, `train`, `evaluate` and `predict` (see the below detailed description for each procedure). OTHER\_ARGS are the specific arguments, which can be found in `paddle/args.py`. 

In the examples below, we use the demo dataset (under `data/demo`) by default to demonstrate the usages of `paddle/run.sh`. 

#### Environment Requirements
Please note that we only tested the baseline on PaddlePaddle v1.2 (Fluid). To install PaddlePaddle, please see [PaddlePaddle Homepage](http://paddlepaddle.org) for more information.

#### Paragraph Extraction
We incorporate a new strategy of paragraph extraction to improve the model performance. The details have been noted in `paddle/UPDATES.md`. Please run the following command to apply the new strategy of paragraph extraction on each document:

```
sh run.sh --para_extraction
```

Note that the preprocessed data should be ready before running this command (see the "Preprocess the Data" section above). The results of paragraph extraction will be saved in `data/extracted/`.

#### Vocabulary Preparation

Before training the model, you need to prepare the vocabulary for the dataset and create the folders that will be used for storing the models and the results. You can run the following command for the preparation:

```
sh run.sh --prepare
```
The above command uses the data in `data/demo/` by default. To change the data folder, you need to specify the following arguments:

```
sh run.sh --prepare --train_files ../data/extracted/train.json --dev_files ../data/extracted/dev.json --test_files ../data/extracted/test.json
```

#### Training

To train a model, please run the following command:

```
sh run.sh --train --pass_num 5
```
This will start the training process with 5 epochs. The trained model will be evaluated automatically after each epoch, and a folder named by the epoch ID will be created under the folder `data/models`, in which the model parameters are saved. If you need to change the default hyper-parameters, e.g. initial learning rate and hidden size, please run the commands with the specific arguments. 

```
sh run.sh --train --pass_num 5 --learning_rate 0.00001 --hidden_size 100
```

More arguments can be found in `paddle/args.py`.


#### Evaluate
To evaluate a specific model, please run the following command:

```
sh run.sh --evaluate  --load_dir YOUR_MODEL_DIR
```
The model under `YOUR_MODEL_DIR` (e.g. `../data/models/1`) will be loaded and evaluated.

#### Inference (Prediction)
To do inference on the test dataset by using a trained model, please run: 

```
sh run.sh --predict  --load_dir YOUR_MODEL_DIR 
```
The predicted answers will be saved in the folder `data/results`.



#### Submit the test results
Once you train a model that is tuned on the dev set, we highly recommend you submit the predictions on test set to the site of DuReader for evaluation purpose. To get inference file on test set:

1. make sure the training is over.
2. select the best model under `data/models` according to the training log.
3. predict the results on test set.
4. [submit the prediction result file](http://ai.baidu.com/broad/submission?dataset=dureader).

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
python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json 
```

By default, the results are saved at `../data/results/` folder. You can change this by specifying `--result_dir DIR_PATH`.

## Run baseline systems on multilingual datasets

To help evaluate the system performance on multilingual datasets, we provide scripts to convert MS MARCO V2 data from its format to DuReader format. 

[MS MARCO](http://www.msmarco.org/dataset.aspx) (Microsoft Machine Reading Comprehension) is an English dataset focused on machine reading comprehension and question answering. The design of MS MARCO and DuReader is similar. It is worthwhile examining the MRC systems on both Chinese (DuReader) and English (MS MARCO) datasets. 

You can download MS MARCO V2 data, and run the following scripts to convert the data from MS MARCO V2 format to DuReader format. Then, you can run and evaluate our DuReader baselines or your DuReader systems on MS MARCO data. 

```
./run_marco2dureader_preprocess.sh ../data/marco/train_v2.1.json ../data/marco/train_v2.1_dureaderformat.json
./run_marco2dureader_preprocess.sh ../data/marco/dev_v2.1.json ../data/marco/dev_v2.1_dureaderformat.json
```

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
