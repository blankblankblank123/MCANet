# MCANet_CVPR2020_submit
Implementation of Multi Co-Attention Network(MCANet) to solve Object Co-segmentation.

### Table of Contents
- <a href='#Installation'>Installation</a>
- <a href='#Datasets'>Datasets</a>
- <a href='#Model'>Model</a>
- <a href='#Demos'>Demos</a>
- <a href='#Evaluation'>Evaluation</a>
- <a href='#Performance'>Performance</a>
- <a href='#Todo'>Todo</a>
&nbsp;
&nbsp;
## Installation
- Clone this repository.
- Python 2.7
- [PyTorch](http://pytorch.org/) 0.4.1 
- Packages: `numpy`,`opencv-python`,`matplotlib`,`glob`
- Datasets: follow the [instructions](#Datasets) below
## Datasets
Four OCS benchmark datasets are used for evaluating the performance of our proposed MACNet.

### VOC10
Collected by Faktor and Irani([paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Faktor_Co-segmentation_by_Composition_2013_ICCV_paper.pdf)), this dataset contains 1,037 images of 20 object classes, and it is the most challenging dataset in the experiment due to extreme intra-class variability, subtle foreground/background discrimination, and class imbalance. This dataset can be downloaded [here](123).
### Internet
Introduced by [paper](http://people.csail.mit.edu/mrub/ObjectDiscovery/), this dataset encloses images with three common object classes. This dataset raises the particular challenge of noisy outliers, such that images with no common object exist in the dataset. This dataset can be downloaded [here](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).
### MSRC
This dataset encompasses 14 image groups with 410 images, each group contains common objects drawn from the same class. This dataset can be downloaded [here](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).
### iCoseg
There are 38 categories in [iCoseg dataset](https://www.cc.gatech.edu/~dbatra/papers/bkpcl_cvpr10.pdf) with totally 643 images. Each category composes images containing foreground with limited intra-class variability.This dataset can be downloaded [here](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).

## Model
<img align="left" src= "https://github.com/blankblankblank123/MCANet_CVPR2020_submit/blob/master/doc/model.PNG">
- Implementation [here](123)
- the codes of resnet101 and ASPP layer are copied from [here](https://github.com/kazuto1011/deeplab-pytorch)
- download pretrained MCANet weights at 123
## Demos
```
python demo.py
```
Input with multi-classes images in 123, output with grouped images. The result can be seen in 123.
## Evaluation
Metric:
- Jaccard
- Precision

```
usage: python 
```

## Performance

## Todo
- Release training code
- Release the test code on iCoseg
- Optimize code comments
