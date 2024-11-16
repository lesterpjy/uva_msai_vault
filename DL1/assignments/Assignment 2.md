## Question 1.1

### a. What is the expression of $f_{1,1}$ according to the above Figure 1? What do you think is missing and how can we mitigate this problem?

$f_{1,1}$ can be constructed by centering the convolution $h_{0,0}$ over the pixel $g_{1,1}$ in image $g$. The equation for obtaining $f_{1,1}$ is thus,
$$
\begin{equation}
\begin{aligned}
f_{1,1} &= g_{2,2}h_{-,-} + g_{2,1}h_{-,0} + g_{2,0}h_{-,+} + g_{1,2}h_{0,-} \\ &+ g_{1,1}h_{0,0} \\&+ g_{1,0}h_{0,+} + g_{0,2}h_{+,-} + g_{0,1}h_{+,0} + g_{0,0}h_{+,+}
\end{aligned}
\end{equation}
$$
However, some of these pixel values are out of  bound for the image $g$, specifically the pixels $g_{2,0}, g_{1,0}, g_{0,2}, g_{0,1}, g_{0,0}$. These pixels are necessary to calculate the convolution $g \circledast h$ centered at $g_{1,1}$. Typical method to mitigate this problem is using padding of the image $g$. A padding method like zero padding simple assign 0 to these padding pixels.

### b. i) Run the given code with net_type=‘Net1’, and vary different conv_type from ‘valid’, ‘replicate’, ‘reflect’, ‘circular’, ‘sconv’ and ‘fconv’, and report the validation and test scores in the form of a table.

 Type of convolution :  valid
 Type of network :  Net1
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.3, 1: 0.1, 2: 0.0, 3: 0.1, 4: 0.0, 5: 0.0, 6: 0.1, 7: 0.3, 8: 0.0, 9: 0.0}
mean: 0.0900 std: 0.1136 for test

*******************************************
 Type of convolution :  replicate
 Type of network :  Net1
*******************************************
Results for validation dataset {0: 98.6, 1: 98.1, 2: 98.8, 3: 98.5, 4: 97.1, 5: 98.8, 6: 98.2, 7: 98.0, 8: 98.5, 9: 98.5}
mean: 98.3100 std: 0.4784 for validation
Results for test dataset {0: 92.9, 1: 92.8, 2: 88.8, 3: 94.7, 4: 95.3, 5: 91.7, 6: 96.3, 7: 94.7, 8: 95.8, 9: 95.0}
mean: 93.8000 std: 2.1629 for test

*******************************************
 Type of convolution :  reflect
 Type of network :  Net1
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

*******************************************
 Type of convolution :  circular
 Type of network :  Net1
*******************************************
Results for validation dataset {0: 99.9, 1: 99.9, 2: 99.7, 3: 99.9, 4: 98.9, 5: 99.7, 6: 99.7, 7: 99.5, 8: 99.8, 9: 99.4}
mean: 99.6400 std: 0.2939 for validation
Results for test dataset {0: 81.5, 1: 82.7, 2: 77.4, 3: 86.9, 4: 83.2, 5: 82.6, 6: 86.5, 7: 79.8, 8: 81.1, 9: 84.4}
mean: 82.6100 std: 2.7504 for test

*******************************************
 Type of convolution :  sconv
 Type of network :  Net1
*******************************************
Results for validation dataset {0: 98.8, 1: 99.0, 2: 98.8, 3: 99.1, 4: 99.0, 5: 99.3, 6: 98.7, 7: 98.8, 8: 98.9, 9: 98.5}
mean: 98.8900 std: 0.2119 for validation
Results for test dataset {0: 6.5, 1: 6.9, 2: 6.5, 3: 6.1, 4: 6.2, 5: 6.5, 6: 6.2, 7: 6.1, 8: 6.4, 9: 6.3}
mean: 6.3700 std: 0.2326 for test

*******************************************
 Type of convolution :  fconv
 Type of network :  Net1
*******************************************
Results for validation dataset {0: 88.4, 1: 87.6, 2: 88.6, 3: 89.9, 4: 88.1, 5: 89.9, 6: 85.8, 7: 87.1, 8: 88.2, 9: 87.1}
mean: 88.0700 std: 1.1984 for validation
Results for test dataset {0: 88.4, 1: 88.6, 2: 88.6, 3: 89.9, 4: 89.1, 5: 89.9, 6: 88.9, 7: 88.4, 8: 88.9, 9: 87.8}
mean: 88.8500 std: 0.6249 for test

| convolution type | validation accuracy (mean) | validation accuracy (std) | test accuracy (mean) | test accuracy (std) |
| ---------------- | -------------------------- | ------------------------- | -------------------- | ------------------- |
| valid            | 100.0000                   | 0.0000                    | 0.0900               | 0.1136              |
| replicate        | 98.3100                    | 0.4784                    | 93.8000              | 2.1629              |
| reflect          | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |
| circular         | 99.6400                    | 0.2939                    | 82.6100              | 2.7504              |
| sconv            | 98.8900                    | 0.2119                    | 6.3700               | 0.2326              |
| fconv            | 88.0700                    | 1.1984                    | 88.8500              | 0.6249              |

### b. ii)  Look at the data samples from the train and test sets provided in the README.md. Based on the structure of the images, guess the pattern of class label 0 and class label 1. How do the samples in the train set differ from the ones in the test set?

The pattern for class label 0 appears to have a red small rectangle on the left of the green small rectangle. The pattern for class label 1 is having the green small rectangle to the left of the red small rectangle.

The test set distribution appears to differ from the train set distribution in that the spatial positions of the rectangles are restricted to different regions of the images. In the training data for label 0, the small rectangles appear in the upper regions of the image, but for the same label 0 in the test set the rectangles appear in the lower region of the image. The opposite is true for the images labeled label 1.


### c. i) From the network architecture of ‘Net1’, infer the variables that affect the conv_type. What is the difference between conv_type ‘valid’, ‘sconv’ and ‘fconv’?

For Net1, the variables that affect the type of convolutions used in the network architecture are the padding type and padding size used for the convolutional layers. The conv_type 'valid', 'sconv', and 'fconv' all use zero padding for the convolutions, but increases in the padding size. 'valid' uses zero padding but pad for a size of 0, which equates to not padding. 'sconv' uses zero padding for a pad size of 1, and 'fconv' uses zero padding for a pad size of 2.

### c. ii) Why do the test accuracies of conv_type=‘valid’, ‘sconv’ and ‘fconv’ (i.e., acc_valid, acc_sconv and acc_fconv) follow the order – acc_valid < acc_sconv < acc_fconv?

The model architecture of Net1 is composed of 4 convolution layers, an adaptive max pooling layer, and then linear layer that maps the flattened channel values to a dimension of 2 for the 2 target classes. The order of the test accuracies for the different conv types with different padding size relates to the image size, and the size of the output after the 4 convolution layers. We can calculate the size of the outputs to an convolution layer with the following formula,
$$
\text{output size} = \frac{\text{input size} + 2 * \text{padding size} - \text{kernel size}}{\text{stride size}} + 1
$$
For conv_type = 'valid', this means that with the input image size of (32, 32) per channel, the output size after the 4 convolution is only (2, 2) per channel. This means that spatial information is largely lost after the convolutions. As established in (b.ii), the pattern that distinguishes the classes arise from the a combination of spatial and channel information: red rectangles on the left for label 0 and green rectangles on the left for label 1. With spatial information lost, conv_type of 'valid' results in a very poor performance.


