# DeepDocClassifier
## Abstract
This is a reimplementation of the **ICDAR-2015 paper** [deepdocclassifier](https://ieeexplore.ieee.org/document/7333933/). 
The model demonstrated here is [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (pretrained on imagenet) and finetuned for document classification on the Tobacoo-3428 dataset available [here](https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=72). A sample of the training set is as following.
![sample training images](imgs/sample_of_training_set.png)
## Results
The original network from the paper gives the result on the left, we get the one on the right
![their confusion](imgs/their_confusion.png) ![my confusion](imgs/myfigure.jpg)