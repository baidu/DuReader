

# DuReader

**DuReader focus on the benchmarks and models of machine reading comprehension for question answering.**

DuReader:

> `DuReader 2.0`: A new large-scale real-world and human sourced MRC dataset [[Paper]](https://www.aclweb.org/anthology/W18-2605.pdf) [[Code]](https://github.com/baidu/DuReader/tree/master/DuReader-2.0) [[Leaderboard]](https://ai.baidu.com/broad/leaderboard?dataset=dureader) 

> `KT-NET`: A machine reading comprehension (MRC) model which integrates knowledge from knowledge bases (KBs) into pre-trained contextualized representations.   [[Paper]](https://aclanthology.org/P19-1226/) [[Code]](https://github.com/baidu/DuReader/tree/master/ACL2019-KTNET) [[Learderboard]](https://rajpurkar.github.io/SQuAD-explorer/) 

> `D-NET`: A simple pre-training and fine-tuning framework which focused on the generalization of machine reading comprehension (MRC) models.    [[Paper]](https://aclanthology.org/D19-5828/) [[Code]](https://github.com/baidu/DuReader/tree/master/MRQA2019-D-NET) [[Learderboard]](https://mrqa.github.io/2019/shared.html) 


> `DuReader Robust`: A dataset challenging models in (1)over-sensitivity, (2)over-stability and (3)generalization.   [[Paper]](https://arxiv.org/abs/2004.11142) [[Code]](https://github.com/baidu/DuReader/tree/master/DuReader-Robust) [[Learderboard]](https://aistudio.baidu.com/aistudio/competition/detail/49/) 

> `DuReader Yes/No`: A dataset challenging models in opinion polarity judgment. [[Code]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_reading_comprehension/DuReader-yesno) [[Leaderboard]](https://aistudio.baidu.com/aistudio/competition/detail/49/)

> `DuReader Checklist`: A dataset challenging model understanding capabilities in vocabulary, phrase, semantic role, reasoning. [[Code]](https://github.com/baidu/DuReader/tree/master/DuReader-Checklist) [[Leaderboard]](https://aistudio.baidu.com/aistudio/competition/detail/66) 
> `DuQM`: Linguistically Perturbed Natural Questions for Evaluating theRobustness of Question Matching Models.[[Code]](https://github.com/baidu/DuReader/tree/master/DuQM) [[Leaderboard]](https://aistudio.baidu.com/aistudio/competition/detail/116)



`DuReader Robust`, `DuReader Yes/No`, `DuReader Checklist` can be downloaded at [qianyan official website](https://www.luge.ai/). `DuReader 2.0` can be downloaded by following the method in `DuReader-2.0/README.md` at this repository.



# News

- June 2021, DuReader Robust, DuReader Yes/No and DuReader Checklist were included in [qianyan](https://www.luge.ai/).
- May 2021, [DuReader Robust](https://arxiv.org/abs/2004.11142) (short paper) was accepted by ACL 2021.
- March 2021, DuReader Checklist was released, holding the [DuReader Checklist challenge](https://aistudio.baidu.com/aistudio/competition/detail/66?isFromLuge=true). 
- March 2020, DuReader Robust was released, holding the [DuReader Robust challenge](https://aistudio.baidu.com/aistudio/competition/detail/28?isFromCcf=true). 
- December 2019, DuReader Yes/No was released, holding the [ DuReader Yes/No challenge](https://ai.xm.gov.cn/competition/project-detail.html?id=1aedc41540e440a59f86a4c543635f64&competeId=0000075d26e840b1b9ffd10633d6a9bf). After that, DuReader Yes/No [Individual Challenge](https://aistudio.baidu.com/aistudio/competition/detail/25) and [Team Challenge](https://aistudio.baidu.com/aistudio/competition/detail/26) were held.
- August 2019, D-NET was released and ranked at top 1 of the [MRQA-2019 shared task](https://mrqa.github.io/2019/shared.html).
- July 2019, [KT-NET](https://aclanthology.org/P19-1226/) was accepted by ACL 2019.
- March 2019, the second MRC challenge was held based on DuReader 2.0, including hard samples in the test set.
- April 2018, [DuReader 2.0](https://www.aclweb.org/anthology/W18-2605.pdf) was accepted by ACL 2018 at the Workshop on Machine Reading for Question Answering.
- March 2018, the [ first MRC challenge]((https://aistudio.baidu.com/aistudio/competition/detail/1).) was held based on DuReader 2.0


# Detailed Description

DuReader contains four datasets: `DuReader 2.0`, `DuReader Robust`, `DuReader Yes/No` and `DuReader Checklist`. The main features of these datasets include:

- Real question,  Real article,  Real answer, Real application scenario;
- Rich question types, including entity, number, opinion, etc;
- Various task types, including span-based tasks and classification tasks;
- Rich task challenges, including model retrieval capability, model robustness, model checklist etc. 

### DuReader 2.0 :  Real question,  Real article,  Real answer

 [[Paper]](https://www.aclweb.org/anthology/W18-2605.pdf) [[Code]](https://github.com/baidu/DuReader/tree/master/DuReader-2.0) [[Leaderboard]](https://ai.baidu.com/broad/leaderboard?dataset=dureader) 

DuReader is a new large-scale real-world and human sourced MRC dataset in Chinese. DuReader focuses on real-world open-domain question answering. The advantages of DuReader over existing datasets are concluded as follows: Real question,  Real article,  Real answer, Real application scenario and Rich annotation.
### KT-NET: Integrate knowledge into pre-trained LMs.  
[[Paper]](https://aclanthology.org/P19-1226/) [[Code]](https://github.com/baidu/DuReader/tree/master/ACL2019-KTNET) [[Learderboard]](https://rajpurkar.github.io/SQuAD-explorer/) 

KT-NET (Knowledge and Text fusion NET) is a machine reading comprehension (MRC) model which integrates knowledge from knowledge bases (KBs) into pre-trained contextualized representations. The model is proposed in ACL2019 paper Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension.

### D-NET: Model generalization
[[Paper]](https://aclanthology.org/D19-5828/) [[Code]](https://github.com/baidu/DuReader/tree/master/MRQA2019-D-NET) [[Learderboard]](https://mrqa.github.io/2019/shared.html)

D-NET is a simple system Baidu submitted for MRQA (Machine Reading for Question Answering) 2019 Shared Task that focused on generalization of machine reading comprehension (MRC) models. The system is built on a framework of pretraining and fine-tuning. The techniques of pre-trained language models and multi-task learning are explored to improve the generalization of MRC models. D-NET is ranked at top 1 of all the participants in terms of averaged F1 score. 
### DuReader Robust: Model Robustness

 [[Paper]](https://arxiv.org/abs/2004.11142) [[Code]](https://github.com/baidu/DuReader/tree/master/DuReader-Robust) [[Learderboard]](https://aistudio.baidu.com/aistudio/competition/detail/49/) 

DuReader Robust is designed to challenge MRC models from the following aspects: (1) over-sensitivity, (2) over-stability and (3) generalization. Besides, DuReader Robust has another advantage over previous datasets: questions and documents are from Baidu Search. It presents the robustness issues of MRC models when applying them to real-world scenarios.

### DuReader Yes/No: Opinion Yes/No Questions

 [[Code]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_reading_comprehension/DuReader-yesno) [[Leaderboard]](https://aistudio.baidu.com/aistudio/competition/detail/49/) 

Span-based MRC tasks adopt F1 and EM metrics to measure the difference between predicted answers and labeled answers. However,  the task about opinion polarity cannot be well measured by these metrics. DuReader Yes/No is proposed to  challenge MRC models in opinion polarity, which will complement the disadvantages of existing MRC tasks and evaluate the effectiveness of existing models more reasonably.

### DuReader Checklist: Natural Language Understanding Capabilities

 [[Code]](https://github.com/baidu/DuReader/tree/master/DuReader-Checklist) [[Leaderboard]](https://aistudio.baidu.com/aistudio/competition/detail/66) 

DuReader Checklist is a high-quality Chinese machine reading comprehension dataset for real application scenarios. It is designed to challenge the natural language understanding capabilities from multi-aspect via systematic evaluation (i.e. checklist), including understanding of vocabulary, phrase, semantic role, reasoning and so on.




# Dataset and Evaluation Tools

We make public a dataset loading and evaluation tool named `qianyan`. You can use this package easily by following the [qianyan repo](https://console.cloud.baidu-int.com/devops/icode/repos/baidu/personal-code/qianyan/tree/master).



# Copyright and License

Copyright 2017 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



# Contact Information

For help or issues using DuReader, including datasets and baselines, please submit a Github issue.

For other communication or cooperation, please contact Jing Liu (`liujing46@baidu.com`) or Hongyu Li (`lihongyu04@baidu.com`).



