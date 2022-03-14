# FastFlow-AD

![image](https://user-images.githubusercontent.com/37028633/157865763-105ae80a-ba4a-4bd0-8bd2-bf0ec392f865.png)


Unoficial implementation of FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows 
https://arxiv.org/pdf/2111.07677v2.pdf

WIP repo. This code has lot of effort to try solving the implementation of this study, which is until unfinished. 
Welcome and appreciate any contributions to the Q&A issue.

The next table shows a quick performance of this code. Please, make sure that your contribution improves the table.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image classification AUROC

|                | Avg  | Carpet | Grid | Leather | Tile | Wood | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| -------------- |:----:|:------:|:----:| :-----: |:----:|:----:|:------:|:-----:|:-------:|:--------:|:---------:|:----:|:-----:|:----------:|:----------:|:------:|
| Wide ResNet-50 | TODO |  0.99  | 0.97 |   1.0   | 1.0  | 1.0  |  1.0   | 0.95  |  0.80   |   0.99   |   0.97    | 0.95 | 0.99  |    0.86    |    0.88    |  1.0   |

Note: We had this results with the "First commit"

### Image segmentation AUROC

TODO

### How to use


```python main.py --dataset {root_path} --classname {classname}```

Example:

```python main.py --dataset ./MVTec --classname bottle```
