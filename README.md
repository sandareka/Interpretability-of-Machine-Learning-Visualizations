# Interpretability-of-Machine-Learning---Visualizations

## Gradient-weighted Class Activation Mapping (Grad-CAM) 

Pycaffe implementation of the paper [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

For this implementation I'm using a pretrained image classification model downloaded from the community in [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo).

For this example, I will use BVLC reference caffenet model which is trained to classify images into 1000 classes. To download the model, go to the folder where you installed Caffe, e.g. C:\Caffe and run
```
 ./scripts/download_model_binary.py models/bvlc_reference_caffenet
 
./data/ilsvrc12/get_ilsvrc_aux.sh
```
![](https://github.com/sandareka/Interpretability-of-Machine-Learning---Visualizations/blob/master/images/cat.jpg =250x250)

Original Image                                                   |  GradCAM
:---------------------------------------------------------------:|:-------------------------:
![test|50%](/images/cat.jpg)  |  ![test](/results/catgradCAM.png)

