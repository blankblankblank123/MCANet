# MCANet_CVPR2020
Implementation of Multi Co-Attention Network(MCANet) to solve Object Co-segmentation.

### Table of Contents
- <a href='#Installation'>Installation</a>
- <a href='#Datasets'>Datasets</a>
- <a href='#Model'>Model</a>
- <a href='#Demos'>Demos</a>
- <a href='#Evaluation'>Evaluation</a>
- <a href='#Performance'>Performance</a>
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
Collected by Faktor and Irani([paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Faktor_Co-segmentation_by_Composition_2013_ICCV_paper.pdf)), this dataset contains 1,037 images of 20 object classes, and it is the most challenging dataset in the experiment due to extreme intra-class variability, subtle foreground/background discrimination, and class imbalance. [Download Link](https://drive.google.com/open?id=1V7YRZafySYOPtZ4WiwqM6pUnjc_8TAH7).
### Internet
Introduced by [paper](http://people.csail.mit.edu/mrub/ObjectDiscovery/), this dataset encloses images with three common object classes. This dataset raises the particular challenge of noisy outliers, such that images with no common object exist in the dataset. [Download Link](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).
### MSRC
This dataset encompasses 14 image groups with 410 images, each group contains common objects drawn from the same class. [Download Link](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).
### iCoseg
There are 38 categories in [iCoseg dataset](https://www.cc.gatech.edu/~dbatra/papers/bkpcl_cvpr10.pdf) with totally 643 images. Each category composes images containing foreground with limited intra-class variability. [Download Link](http://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip).

## Model
   
- [Implementation](https://github.com/blankblankblank123/MCANet_CVPR2020_submit/tree/master/libs/models)
- the codes of resnet101 and ASPP layer are copied from [this work](https://github.com/kazuto1011/deeplab-pytorch)
- download pretrained MCANet weights at [here](https://drive.google.com/open?id=1tyM2tJ_LhfCmI3rsliciLidfHk5ploKc)

<img align="left" src= "https://github.com/blankblankblank123/MCANet_CVPR2020_submit/blob/master/doc/model.PNG">

## Demos
```
Usage: python demo.py
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

Jaccard index reports the ratio of the intersection and union area between the segmented foreground objects and the ground truth.
- Precision

Precision measures the percentage of correctly segmented object and background pixels.
```
Usage: python eval_vis_voc_group.py | eval_vis_internet_group.py | eval_vis_msrc_group.py
```

## Performance
- average
<table>
    <tr>
        <th>Dataset</th>
        <td colspan="2">VOC10</td>
        <td colspan="2">Internet</td>
        <td colspan="2">MSRC</td>
    </tr>
    <tr>
        <th> Metric</th>
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
        <th>63.7</th>
        <th>94.2</th>
        <th>74.8</th>
        <th>90.1</th>
        <th>73.5</th>
    </tr>
   
</table>

- voc10
<table>
    <tr>
        <th> Class</th>
        <th colspan="2">A.P.</th>
        <th colspan="2">Bike.</th>
        <th colspan="2">Bird</th>
        <th colspan="2">Boat</th>
        <th colspan="2">Bottle</th>
        <th colspan="2">Bus.</th>
        <th colspan="2">Car</th>
        <th colspan="2">Cat</th>
        <th colspan="2">Chair</th>
        <th colspan="2">Cow</th>
        <th colspan="2">D.T.</th>
        <th colspan="2">Dog</th>
        <th colspan="2">Horse</th>
        <th colspan="2">M.B.</th>
        <th colspan="2">P.S.</th>
        <th colspan="2">P.P.</th>
        <th colspan="2">Sheep</th>
        <th colspan="2">Sofa</th>
        <th colspan="2">Train</th>
        <th colspan="2">TV</th>
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
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
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
        <th>97.2</th>
        <th>77.9</th>
        <th>90.7</th>
        <th>13.1</th>
        <th>96.6</th>
        <th>73.1</th>
        <th>95.9</th>
        <th>71.8</th>
        <th>94.6</th>
        <th>72.1</th>
        <th>94.5</th>
        <th>84.2</th>
        <th>95.3</th>
        <th>82.0</th>
        <th>95.0</th>
        <th>77.8</th>
        <th>89.7</th>
        <th>39.2</th>
        <th>95.8</th>
        <th>78.4</th>
        <th>86.6</th>
        <th>15.7</th>
        <th>95.3</th>
        <th>72.1</th>
        <th>94.7</th>
        <th>73.3</th>
        <th>92.9</th>
        <th>69.7</th>
        <th>93.7</th>
        <th>56.1</th>
        <th>92.1</th>
        <th>51.5</th>
        <th>93.7</th>
        <th>72.3</th>
        <th>88.8</th>
        <th>49.5</th>
        <th>95.3</th>
        <th>79.6</th>
        <th>91.6</th>
        <th>64.5</th>
    </tr>
   
</table>

- internet
<table>
    <tr>
        <th> Class</th>
        <th colspan="2">Airplane</th>
        <th colspan="2">Car</th>
        <th colspan="2">Horse</th>
    </tr>
    <tr>
        <th> Metric</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
    </tr>
       <tr>
        <th> MCANet</th>
        <th>96.1</th>
        <th>73.2</th>
        <th>94.7</th>
        <th>84.2</th>
        <th>91.8</th>
        <th>67.1</th>
    </tr>
   </table>

- MSRC
<table>
    <tr>
        <th> Class</th>
        <th colspan="2">house</th>
        <th colspan="2">plane</th>
        <th colspan="2">bike</th>
        <th colspan="2">bird</th>
        <th colspan="2">dog</th>
        <th colspan="2">sign</th>
        <th colspan="2">tree</th>
        <th colspan="2">chair</th>
        <th colspan="2">face</th>
        <th colspan="2">flower</th>
        <th colspan="2">car</th>
        <th colspan="2">sheep</th>
        <th colspan="2">cow</th>
        <th colspan="2">cat</th>
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
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
        <th>P</th>
        <th>J</th>
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
        <th>90.6</th>
        <th>78.1</th>
        <th>94.6</th>
        <th>74.3</th>
        <th>75.4</th>
        <th>49.8</th>
        <th>96.5</th>
        <th>78.5</th>
        <th>95.6</th>
        <th>82.5</th>
        <th>95.0</th>
        <th>86.4</th>
        <th>66.7</th>
        <th>35.4</th>
        <th>90.8</th>
        <th>67.5</th>
        <th>87.7</th>
        <th>67.7</th>
        <th>91.8</th>
        <th>80.3</th>
        <th>91.5</th>
        <th>81.2</th>
        <th>95.4</th>
        <th>84.3</th>
        <th>94.6</th>
        <th>80.7</th>
        <th>95.2</th>
        <th>82.5</th>
    </tr>
   </table>
