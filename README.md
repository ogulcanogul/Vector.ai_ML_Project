# vectorAI_ML_Project

This is a multi-class image classifier on fashion MNIST dataset using a residual neural network (ResNet). 

## Get Started
Run
```
pip install -r requirements.txt
```

## Custom Dataset

You can use your own dataset class in ```/framework/datasets``` folder by overriding _reOrganizeDataset method 
of Dataset class implemented in ```/framework/datasets/dataset_base.py```. You can find an example code in 
```/framework/datasets/fashion_mnist.py```

## Config Files

You change network/optimizer parameters and the description of classes in your own dataset from ```/configs/config.yaml```.

## Training

Run
```
train.py
```

## Run Flask App

Run
```
app.py
```
Then, you can go to ```http://127.0.0.1:5000/``` to upload images and see the results.

## Author

Oğul Can

ogulcanogul@gmail.com
