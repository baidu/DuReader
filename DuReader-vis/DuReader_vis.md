# DuReader<sub>vis</sub>
This is a repository of the paper:[DuReader<sub>vis</sub>,: A Chinese Dataset for Open-domain Document
Visual Question Answering](https://aclanthology.org/2022.findings-acl.105.pdf) ACL 2022 Findings.

# Introduction

![Figure 1: Comparison between Open Domain QA and Open Domain DocVQA](images/intro.png)

Open-domain question answering (Open-domain QA) ( Figure 1(a)) has been used in a wide range of applications, such as web search and enterprise search, which usually takes clean texts extracted from various formats of documents (e.g., web pages, PDFs, or Word documents) as the information source. However, designing different text extraction approaches is time-consuming and not scalable, where a scalable QA system should process various formats of documents at a low cost, and not be restricted by the document format. 

To tackle the above limitations, we propose and study an **Open-domain Document Visual Question Answering (Open-domain DocVQA)** task ( Figure 1(b)),  which requires answering questions based on a collection of document images directly instead of only document texts, utilizing layouts and visual features additionally. In this task, we apply a universal document extractor (e.g., OCR) to extract all the texts and layouts from the document images and then utilize them along with the visual features to perform the following procedures, including **Document Visual Retrieval (DocVRE)** to retrieve relevant document images, and **Document Visual Question Answering (DocVQA)** to extract answers from retrieved document images.

To advance this task, we create the first Chinese Open-domain DocVQA dataset called DuReader<sub>vis</sub>, containing about 15K question-answering pairs and 158K document images from the Baidu search engine. The questions are real ones issued by users to the search engine. Besides, the document images are converted from web pages that are easy to obtain with long documents, complex layouts, and rich visual features. In addition, the answers in DuReader<sub>vis</sub> contain long answers, such as multi-span texts, lists, and tables. 

There are three main challenges in DuReader<sub>vis</sub>: (1) long document understanding, (2) noisy texts, and (3) multi-span answer extraction. 

## Dataset

The Open-domain DocVQA task consists of two stages: DocVRE and DocVQA. We list all the dataset in DuReader<sub>vis</sub> for both stages in the following table:

| FileName                                                     | MD5                              | Description                                                  |
| ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| [dureader_vis_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_docvqa.tar.gz) | 03559a8d01b3939020c71d4fec250926 | The train and dev dataset for DocVQA. We align the textual answer to the OCR results of documents, tokenize the OCR results by the LayoutXLM tokenizer, and generate the label sequence for training. |
| [dureader_vis_open_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_open_docvqa.tar.gz) | 5907ce4126d3eef8ca32d291dbf14abb | (1) The original dataset for open-domain DocVQA. and (2) Top-1 document image retrieved by BM25. |
| [dureader_vis_ocr.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_ocr.tar.gz) | 48d17330bc301cd8966d97d954d33853 | The OCR results of all 158K images.                          |
| [dureader_vis_images_part_1.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_1.tar.gz) | 6f41c5efe457f8acd35de8599e083c89 | Original image part 1                                        |
| [dureader_vis_images_part_2.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_2.tar.gz) | 6cb6600095aae1e625351bc006bcc906 | Original image part 2                                        |
| [dureader_vis_images_part_3.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_3.tar.gz) | 00a616c2421ce30a8e1d106f24fe78db | Original image part 3                                        |
| [dureader_vis_images_part_4.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_4.tar.gz) | 3a8ba1a5bb7c8abbd25a45c1daf9aa85 | Original image part 4                                        |
| [dureader_vis_images_part_5.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_5.tar.gz) | 920af983ccf39a74f4d438c8d43549f5 | Original image part 5                                        |
| [dureader_vis_images_part_6.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_6.tar.gz) | a671f142e55f26888cfb965010d88e8c | Original image part 6                                        |
| [dureader_vis_images_part_7.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_7.tar.gz) | cb53d8a0f17a2a0f8a0791634cf35d96 | Original image part 7                                        |
| [dureader_vis_images_part_8.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_8.tar.gz) | 3046edb565d90ecb385d5c44430ccc60 | Original image part 8                                        |
| [dureader_vis_images_part_9.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_9.tar.gz) | 51a8ccf2cce9ef6b614045cac99b2526 | Original image part 9                                        |
| [dureader_vis_images_part_10.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_images_part_10.tar.gz) | 73e9f4282b0a8d432df9fc4a79627134 | Original image part 10                                       |



If you focus on the DocVQA task, dataset [dureader_vis_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_docvqa.tar.gz) should be downloaded

If you focus on the Open Domain DocVQA task, dataset [dureader_vis_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_docvqa.tar.gz) and [dureader_vis_open_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_open_docvqa.tar.gz) should be downloaded.

If you would like to process the dataset using the original OCR results, dataset [dureader_vis_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_docvqa.tar.gz) , [dureader_vis_open_docvqa.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_open_docvqa.tar.gz)  and [dureader_vis_ocr.tar.gz](https://dataset-bj.cdn.bcebos.com/qianyan/dureader_vis_ocr.tar.gz) should be downloaded.

If you would like to start from the initial point, all the datasets should be downloaded.




# DuReader<sub>vis</sub> Baseline System
The baseline code will come soon...

# Citation

If you find our paper and code useful, please cite the following paper:

```latex
@inproceedings{dureadervis2022acl,
  title={DuReader\({}_{\mbox{vis}}\): {A} Chinese Dataset for Open-domain Document Visual Question Answering},
  author={Le Qi, Shangwen Lv, Hongyu Li, Jing Liu, Yu Zhang, Qiaoqiao She, Hua Wu, Haifeng Wang and Ting Liu},
  booktitle={Findings of the Association for Computational Linguistics: {ACL} 2022,
               Dublin, Ireland, May 22-27, 2022},
  pages={1338--1351},
  year={2022}
}
```




# Copyright and License
Copyright 2022 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.





