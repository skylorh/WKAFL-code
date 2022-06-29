# WKAFL
The projects are deployed on four datasets under PySyft framework. The four datasets are CelebA, EMNIST ByClass, EMNIST MNIST and CIFAR10. The first two benchmark FL datasets provided by LEAF were directly used. 

To run the files, you must replace the dataloader.py of PySyft framework under Lib\site-packages\syft\frameworks\torch\fl with the dataloader.py we provide.

```
conda create -n py37 python==3.7
conda activate py37
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip3 install syft==0.2.3
```