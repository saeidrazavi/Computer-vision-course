in this HW ,  we want to train cnn models with different architectures for scene recognition . below you can see these structures : 

• PART 1 : using first convolutional layer and two last fully connected layer of Alexnet
architecture . note that we use max-pooling layer with kernel-size=4 and stride=4 in first layer .

• PART 2 : using three first convolutions layer and three last fully connected layer of Alexnet
architecture . in the third layer , we change the conv2d layer with depth of 384 to 256 .

• PART 3 : using whole Alexnet architecture and just change the last fully-connected layer with
depth of 1000 to 15 but without any pre-trained weights. more specifically , we don’t use
pre-trained parameters that derived for architecture from imagenet dataset .

• PART 4 : using whole Alexnet architecture and just change the last fully-connected layer with
depth of 1000 to 15 . we use pre-trained weights for all the layers expect for new classifier layer
.during training , we just let the parameters of new classifier layer to update in each epoch and
freeze all other parameters.

• PART 5 : using whole Alexnet architecture and just change the last fully-connected layer with
depth of 1000 to 15 . we use pre-trained weights for all the layers expect for new classifier layer
.during training , we let all parameters of all layers to update in each epoch
