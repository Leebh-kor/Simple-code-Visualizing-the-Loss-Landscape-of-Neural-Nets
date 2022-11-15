# Simple code to reproduce the paper *Visualizing the Loss Landscape of Neural Nets* with Pytorch 1.8
Reimplmentation of [*Visualizing the Loss Landscape of Neural Nets*](https://github.com/tomgoldstein/loss-landscape) with PyTorch 1.8

There are reasons for reproducing the existing code.
1. Complexity of code
2. Package dependencies of code (e.g., PyTorch 0.4)

The purpose of this repository is to simply use the visualization methods introduced in the [paper](https://arxiv.org/abs/1712.09913) using pytorch 1.8.0

It can be applied to metirc or loss used in your study by simply modifying this repository.

## Requirements
- torch==1.8.0 torchvision 0.9.0 with cuda 11.1
- numpy
- scipy
- h5py
- matplotlib
- seaborn

```python
pip install -r requirements.txt
```

## How to use
The configurations to visualize the loss landscape of ResNet56 on the CIFAR-10 dataset described in the paper are set as default.
```python
sh train.sh
```
When the above code is executed, the 2D-visualization results and vtp file for 3D rendering are saved in (`./result`) path

How to visualize a 3D loss surface by opening a vtp file via paraview is well described in the original [repository](https://github.com/tomgoldstein/loss-landscape).

## 2D visualization of the loss surface
The following Contour Plots are 2D loss surfaces for ResNet56 and ResNet56-no short obtained with default configurations.


<table style="display: inline-table;">
<div align="center">  
<tr><td><img src = "fig\resnet56_surface_file.png" width="700px" height="500px"></td></tr>
<tr><td><div align="center">Loss surface of ResNet56, Test error : 11.03% </td></tr>
</table>

<table style="display: inline-table;">
<div align="center">  
<tr><td><img src = "fig\ns_resnet56_surface_file.png" width="700px" height="500px"></td></tr>
<tr><td><div align="center">Loss surface of ResNet56-NS, Test error : 18.09% </td></tr>
</table>

## 3D visualization of the loss surface
This is a 3D version of the same loss surface as above.
If you set a high resolution, you can get visualization results similar to papers. (In the paper, 251 x 251, we have 51 x 51.)
<table style="display: inline-table;">
<div align="center">  
<tr><td><img src = "fig\resnet56_3D_surface.png" width="700px" height="500px"></td></tr>
<tr><td><div align="center">Loss surface of ResNet56, Test error : 11.03% </td></tr>
</table>

<table style="display: inline-table;">
<div align="center">  
<tr><td><img src = "fig\nsresnet56_3D_surface.png" width="700px" height="500px"></td></tr>
<tr><td><div align="center">Loss surface of ResNet56-NS, Test error : 18.09% </td></tr>
</table>
