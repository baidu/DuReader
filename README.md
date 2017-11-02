# DuReader Dataset

DuReader is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows:
 - Real question
 - Real article
 - Real answer
 - Real application scenario
 - Rich annotation

# DuReader Baseline System

DuReader Baseline System implements 2 classic reading comprehension models([BiDAF]() and [Match-LSTM]()) on [DuReader dataset](). The system is implemented with 2 frameworks: [PaddlePaddle]() and [TensorFlow]().

## How to Run

### Download Dataset

To Download DuReader dataset:

```bash
cd data && bash download.sh
```
For more details about DuReader dataset please refer to [DuReader Homepage]().

### Data Preparation

After the dataset is downloaded, there is still some work to do to run the baseline systems. DuReader dataset offers rich amount of documents for every user question, the documents are too long for popular RC models to cope with. In our baseline models, we preprocess the train set and development set data by selecting the paragraph that is most related to the answer string, while for inferring(no available golden answer), we select the paragraph that is most related to the question string. The preprocessing strategy is implemented in `utils/preprocess.py`. To preprocess the raw data, run:

```bash
cat data/raw/search.train.json | python utils/preprocess.py > data/preprocessed/search.train.json
```

The preprocessing is already included in `data/download.sh`, the preprocessed data is stored in `data/preprocess`, the downloaded raw data is under `data/raw`.

### PaddlePaddle
#### Environment Requirements
Install the latest PaddlePaddle by:

```bash
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

To install PaddlePaddle by other ways and for more details about PaddlePaddle, see [PaddlePaddle Homepage]().

#### Training

We implement 3 models with PaddlePaddle: Match-LSTM, BiDAF, and a classification model for data with `query_type='YES_NO'`, the model simply replaces the Pointer-Net on top of Match-LSTM model with a one-layered classifier. The 3 implemented models can all be trained and inferred by run `run.py`, to specify the model to train or to infer, use `--algo [mlstm|bidaf|yesno]`, for complete usage run `python run.py -h`.

The basic training and inference process has been wrapped in `run.sh`,  the basic usage is:

```
bash run.sh EXPERIMENT_NAME ALGO_NAME TASK_NAME
```

`EXPERIMENT_NAME` can be any legal folder name,  `ALGO_NAME` should be `bidaf`, `mlstm` or `yesno` for the 3 models have been implemented.
For example, to train a model with BiDAF, run:

```bash
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

#### Inference

To infer a trained model, run the same command as training and change `train` to `infer`,  and add `--testset <path_to_testset>` argument. for example, suppose the 'test_bidaf' experiment is successfully trained,  to infer the saved models, run:

```
bash run.sh test_bidaf bidaf infer --testset ../data/preprocessed/search.test.json
```

The results corresponding to each model saved is under `infer` folder, and the evaluation metrics is logged into the infer log files under `log`.
#### Test Result Submission
You can infer and evaluate your models on development data set locally by following the above steps, once you've developed a model that works to your expectation on the dev set, we highly recommend you to submit your inference result on the released test set to us to evaluate. To get inference file on test set:

1. make sure the training is over.
2. infer your models on dev set and pick the best model.
3. keep only the best model under `models/<EXPERIMENT_NAME>/models`.
4. infer again with test set.
5. [submit the infer result file]().

### Tensorflow

#### Environment Requirements

#### Training

#### Inference

#### Result Evaluation

#### Test Result Submission

## Copyright and License
