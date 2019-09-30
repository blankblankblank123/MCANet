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
Collected by Faktor and Irani([paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Faktor_Co-segmentation_by_Composition_2013_ICCV_paper.pdf)), this dataset contains 1,037 images of 20 object classes, and it is the most challenging dataset in the experiment due to extreme intra-class variability, subtle foreground/background discrimination, and class imbalance. [Download Link](123).
### Internet
Introduced by [paper](http://people.csail.mit.edu/mrub/ObjectDiscovery/), this dataset encloses images with three common object classes. This dataset raises the particular challenge of noisy outliers, such that images with no common object exist in the dataset. [Download Link](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).
### MSRC
This dataset encompasses 14 image groups with 410 images, each group contains common objects drawn from the same class. [Download Link](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).
### iCoseg
There are 38 categories in [iCoseg dataset](https://www.cc.gatech.edu/~dbatra/papers/bkpcl_cvpr10.pdf) with totally 643 images. Each category composes images containing foreground with limited intra-class variability. [Download Link](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).

## Model
   
- [Implementation](https://github.com/blankblankblank123/MCANet_CVPR2020_submit/tree/master/libs/models)
- the codes of resnet101 and ASPP layer are copied from [this work](https://github.com/kazuto1011/deeplab-pytorch)
- download pretrained MCANet weights at 123

<img align="left" src= "https://github.com/blankblankblank123/MCANet_CVPR2020_submit/blob/master/doc/model.PNG">

## Demos
```
python demo.py
```
Input with multi-classes [demo images](https://github.com/blankblankblank123/MCANet_CVPR2020_submit/tree/master/demo_images), output with grouped images. The result visualization can be seen [here](https://github.com/blankblankblank123/MCANet_CVPR2020_submit/tree/master/result/demo).
group 1:
<img align="left" src= "https://github.com/blankblankblank123/MCANet_CVPR2020_submit/blob/master/result/demo/0.png">
group 2:
<img align="left" src= "https://github.com/blankblankblank123/MCANet_CVPR2020_submit/blob/master/result/demo/1.png">
group 3:
<img align="left" src= "https://github.com/blankblankblank123/MCANet_CVPR2020_submit/blob/master/result/demo/2.png">

## Evaluation
Metric:
- Jaccard
- Precision

```
usage: python 
```

## Performance
<table>
    <tr>
        <th>Dataset</th>
        <td colspan="2">VOC10</td>
        <td colspan="2">Internet</td>
        <td colspan="2">MSRC</td>
        <td colspan="2">iCoseg</td>
    </tr>
    <tr>
        <th> Metric</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
    </tr>
    <tr>
        <th> MCANet</th>
        <th>93.5</th>
        <th>64.2</th>
        <th>93.6</th>
        <th>74.3</th>
        <th>90.5</th>
        <th>74.4</th>
        <th>94.0</th>
        <th>77.2</th>
    </tr>
   
</table>

## Todo
- Release training code
- Release the test code on iCoseg
- Optimize code comments
