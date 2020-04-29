In this model, we are considering 10 different everyday objects and using a built in dataset CIFAR Image Dataset.

A common architecture for a CNN is a stack of Conv2D and MaxPooling2D layers followed by a few denesly connected layers. 
The idea is that the stack of convolutional and maxPooling layers extract the features from the image. 
Then these features are flattened and fed to densly connected layers that determine the class of an image based on the presence of features.

Layer 1:
The input shape of our data will be 32, 32, 3 and we will process 32 filters of size 3x3 over our input data. 
We will also apply the activation function relu to the output of each convolution operation.

Layer 2:
This layer will perform the max pooling operation using 2x2 samples and a stride of 2.

Other Layers:
The next set of layers do very similar things but take as input the feature map from the previous layer. 
They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks in spacial dimensions as it passed through the layers, meaning we can afford (computationally) to add more depth.