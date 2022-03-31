# DuReader-Retrieval

## 任务简介 Task Description
给定一个问题q 及其所有相关段落的集合 Pq，其中p∈Pq 为与q相关的单条段落，以及一个包含所有候选段落的集合 P。参赛系统的目标是根据 q，从P中检索出所有与q相关的段落Pq，并将Pq 中的段落尽可能排序到检索结果列表靠前的位置。数据集中的每个样本为 <q,Pp> ，例如：

Given a query q with the set of all its relevant paragraphs Pq where p∈Pq is a single relevant paragraph of q, and a collection P containing all available paragraphs. The target is to retrieve all p∈Pq from P for each q and rank them as higher as possible in the final ranking list. An example is shown as below:

问题 (q): 太阳花怎么养

篇章1 (p1∈Pq): 花卉名称:太阳花播种时间:春、夏、秋均可播种为一年生肉质草本植物。株高10～15cm。花瓣颜色鲜艳,有白、深黄、红、紫等色。园艺品种很多,有单瓣、半重瓣、重瓣之分。喜温暖、阳光充足而干燥的环境,极耐瘠薄,一般土壤均能适应,能自播繁衍。见阳光花开,早、晚、阴天闭合,故有太阳花、午时花之名。花期6～7月。太阳花种子非常细小。常采用育苗盘播种,极轻微地覆些细粒蛭石,或仅在播种后略压实,以保证足够的湿润。发芽温度21～24℃,约7～10天出苗,幼苗极其细弱,因此如保持较高的温度,小苗生长很快,便能形成较为粗壮、肉质的枝叶。这时小苗可以直接上盆,采用10厘米左右直径的盆,每盆种植2～5株,成活率高,生长迅速。

  
篇章2 (p2∈Pq): 抹平容器中培养土平面,将剪来的太阳花嫩枝头插入竹筷戳成的洞中,深入培养土最多不超过2厘米。为使盆花尽快成形、丰满,一盆中可视花盆大小,只要能保持2厘米的间距,可扦插多株(到成苗拥挤时,可分栽他盆)。接着浇足水即可。新扦插苗可遮阴,也可不遮阴,只要保持一定湿度,一般10天至15天即可成活,进入正常的养护。太阳花极少病虫害。平时保持一定湿度,半月施一次千分之一的磷酸二氢钾,就能达到花大色艳、花开不断的目的。如果一盆中扦插多个品种,各色花齐开一盆,欣赏价值更高。每年霜降节气后(上海地区)将重瓣的太阳花移至室内照到阳光处。入冬后放在玻璃窗内侧,让盆土偏干一点,就能安全越冬。次年清明后,可将花盆置于窗外,如遇寒流来袭,还需入窗内养护。

Question (q): How to raise the sunflower?

Paragraph 1 (p1∈Pq): Flower name: Sunflower Sowing time: Spring, summer, and autumn can be sown as an annual succulent herb. Plant height is 10-15cm. The petals are bright in color, white, dark yellow, red, purple and other colors. There are many horticultural varieties, including single, semi-double and double petals. It likes a warm, sunny and dry environment, extremely tolerant to barrenness, and can adapt to general soils and can reproduce by itself. See the sun flower blooming, morning, evening, cloudy day closed, so it is called the sun flower, noon flower. Flowering from June to July. Sunflower seeds are very small. The seedling trays are often used for sowing, very lightly covered with fine vermiculite, or only slightly compacted after sowing to ensure sufficient moisture. The germination temperature is 21～24℃, and the seedlings emerge in about 7～10 days. The seedlings are extremely thin. Therefore, if the temperature is kept high, the seedlings will grow quickly, and thicker, fleshy branches and leaves can be formed. At this time, the seedlings can be directly put into pots, using pots with a diameter of about 10 cm, planting 2 to 5 plants per pot, with high survival rate and rapid growth.

Paragraph 2 (p2∈Pq): Flatten the soil surface in the container, insert the cut branches of sunflower into the hole made by the bamboo chopsticks, and deepen the soil for no more than 2 cm. In order to make the potted flowers take shape and fullness as soon as possible, the size of the pot can be seen in a pot, as long as the spacing of 2 cm can be maintained, multiple plants can be cut (when the seedlings are crowded, they can be planted in other pots). Then pour plenty of water. The new cuttings can be shaded or not. As long as they maintain a certain humidity, they can survive 10 to 15 days and enter normal maintenance. Sunflower has very few pests and diseases. Maintain a certain humidity at ordinary times, and apply one-thousandth of potassium dihydrogen phosphate once a half month to achieve the purpose of large flowers and continuous blooming. If there are multiple varieties of cuttings in one pot, the flowers of all colors will bloom in one pot, and the appreciation value will be higher. Every year after the frost falls (Shanghai area), the double-flowered sunflowers are moved indoors to shine in the sun. Put it on the inside of the glass window after the winter, and let the potting soil dry a little, so that you can survive the winter safely. After the Qingming Festival in the following year, the flowerpots can be placed outside the window.

## 数据集 Datasets
DuReader_retrieval[1]数据集包含训练集、开发集、测试集和段落语料库：

The DuReader_retrieval dataset includes training set, development set, test set and passage collection:

训练集: 训练集以大规模中文阅读理解数据集DuReader 2.0为基础构建而成，共提供了约8.6W个 <q,Pq> 样本用于模型的训练；数据集中的问题均为百度搜索中的真实用户问题，数据集中的篇章均来自百度搜索结果。

Training set: The training set is constructed using the large-scale Chinese reading comprehension dataset DuReader 2.0. It contains ~86K <q,Pq> samples for model training; The questions in the data set are the real questions issued by users in Baidu search, the paragraphs are extracted from the search results of Baidu search engine.

开发集： 数据构造方法和分布与训练集完全相同，共包含2K个样本，用于模型整体性能调试。

Development set: the construction and the distribution of data is same as the training set, including 2K samples. Used for evaluating overall performance of the model.

测试集： 数据构造方法和分布与训练集完全相同，共包含4K样本，用于模型的最终评测。

Test set: the construction and the distribution of data is same as the training set, including 4K samples, used for final evaluation.

段落语料库： 从真实网页文档中抽取并清理得到，共包含809万段落数据。作为检索的候选来源，包括了训练集、开发集和测试集中的所有段落。

Paragraph collection: There are about 8.09 million passages in total, extracted and cleaned from real web documents, providing the source of paragraphs for the retrieval model.

注：报名后在【数据集介绍页】查看数据集格式，并下载数据集。

Note: After registration, you can check the format of the dataset and download the dataset on the page of Dataset Introduction.

[1] 有关Dureader_retrieval数据集的更多信息可参考论文：《DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine》（https://arxiv.org/abs/2203.10232 ）。如果你对Dureader系列的其他任务数据集感兴趣，可以通过 https://github.com/baidu/DuReader 获取；我们也鼓励选手使用这些数据来提升模型效果。

[1] More information about the Dureader_retrieval dataset can be found in the paper: https://arxiv.org/abs/2203.10232, where you can get more information about the dataset. If you are interested in other task datasets of Dureader series, you can get them through https://github.com/baidu/DuReader. We also encourage contestants to use these data to improve the model effect.

## 评价方法 Evaluation Metrics
本次评测所采用的评价指标为Mean Reciprocal Rank（MRR）和Top-K召回率（Recall@K）。具体来说，我们使用MRR@10、 Recall@1和Recall@50作为评估指标。其中MRR@10作为主要评价指标，系统最终排名根据在测试集中所有样本上的MRR@10平均值（微平均，Micro Average）排序得到。

The evaluation metrics used in this task are Mean Reciprocal Rank (MRR) and Top-K recall (Recall@K). Specifically, we use the Mean Reciprocal Rank (MRR@10) of the top 10 retrieved paragraphs, the recall of the top 1 retrieved paragraphs (Recall@1), and the recall of the top 50 retrieved paragraphs (Recall@50). MRR@10 will be used as the main evaluation metric. The final performance is evaluated on all the samples from the test set (micro-average).

## 基线系统 Baseline
本次评测将提供基于飞桨框架PaddlePaddle的开源基线系统，提供丰富的高层API，从开发、训练到预测部署提供优质的整体体验。推荐您参照基线方案，进行二次开发、模型调优和方案创新。

An open-source baseline system is provided based on PaddlePaddle. With the various high-level APIs, you could have better experience in the whole process of model development, training, inferencing and deployment. It is highly recommend to follow the baseline system to build your models efficiently.

基线系统给出了召回（retrieval）和重排（rerank）两个基线模型，基于先进的RocketQA训练方式进行训练，开源代码地址如下：https://github.com/PaddlePaddle/RocketQA/tree/main/research/DuReader-Retrieval-Baseline。 RocketQA开源代码库中同时提供了简单易用的对偶模型训练和预测工具，以及11个业界领先的预置模型，欢迎关注和使用。

The baseline system provides two baseline models, for retrieval and re-rank, which are trained based on the advanced RocketQA training method. The open source code address is as follows: https://github.com/PaddlePaddle/RocketQA/tree/main/research/DuReader-Retrieval-Baseline The open source repository also provides easy-to-use tools for training and prediction of dual-encoder model , as well as 11 industry-leading preset models. Welcome to pay attention.
