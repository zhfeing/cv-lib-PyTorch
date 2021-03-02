# deep-learning-lib

Utils for deep learning training, evaluating...

## Classification

### Datasets
1. Cifar 10 & 100
2. ImageNet
3. TinyImageNet

### Models

1. ResNet for Cifar
2. MobileNet V2 for Cifar
3. ResNet for ImageNet without some inplace ReLU
4. VGG
5. WRN for Cifar

All models support middle feature extracting, which is crucial for distillation.

## Detection

### Datasets
1. VOC 2007 & 2012
2. COCO 2017

### Models
1. SSD-ResNet
2. SSD-VGG

## Utils

### Distributed Logger

### Metrics
1. Classification
2. Detection Average Precision

### Schedulers
1. `ConstantLR`
2. `PolynomialLR`
3. `WarmUpLR`

### Configure Parsing

### Cuda Utils
1. Preserve Memory

### Distributed Utils

