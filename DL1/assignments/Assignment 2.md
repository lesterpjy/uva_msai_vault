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

The model architecture of Net1 is composed of 4 convolution layers, an adaptive max pooling layer, and then linear layer that maps the maxpooled channel values to a dimension of 2 for the 2 target classes. The order of the test accuracies for the different conv types with different padding size relates to the image size, and the size of the output after the 4 convolution layers. We can calculate the size of the outputs to an convolution layer with the following formula,
$$
\text{output size} = \frac{\text{input size} + 2 * \text{padding size} - \text{kernel size}}{\text{stride size}} + 1
$$
For conv_type = 'valid', this means that with the input image size of (32, 32) per channel, the output size after the 4 convolution is only (2, 2) per channel. This means that spatial information is largely lost after the convolutions, and the max pool layer select only the strongest signal from the (2, 2) for each channel. As established in (b.ii), the pattern that distinguishes the classes arise from the a combination of spatial and channel information: red rectangles on the left for label 0 and green rectangles on the left for label 1. With spatial information nearly all lost, conv_type of 'valid' results in a very poor performance. It can be observed that as the padding size increase to 1 for sconv and 2 for fconv, more spatial information are preserved in the final output. (4, 4) for sconv and (6, 6) for fconv. With fconv we then observe that enough spatial information are preserved such that a higher test accuracy of 88.8 can be achieved.

### c. iii) Why is the test accuracy of conv_type=‘reflect’ less than ‘fconv’?

With conv type of reflect, the padding type used for all 4 convolution layers of the network is 'reflect', which pads the image with pixel values "reflected" from the edges from the image. For example with a pad size of 1 using reflect padding, 
$$
\begin{array}{ccc}
1 &2 &3\\
4 &5 &6 \\
7 &8 &9
\end{array} \ \ \ \ \Rightarrow \ \ \ 
\begin{array}{ccccc}
5 &4 &5 &6 &5 \\
2 &1 &2 &3 &2 \\
5 &4 &5 &6 &5 \\
8 &7 &8 &9 &8 \\
5 &4 &5 &6 &5
\end{array}
$$
We can observe that the model trained with reflect achieves a perfect accuracy on the validation data where label 0 have rectangles red, green (in that order) in the upper region of the image and label 0 have rectangles green, red (in that order) in the lower region of the image. Intuitively, by reflecting the pixels for a pad size of 2, the signals of the small rectangles on their respective sides are reflected to the opposite sides. However, this reflection forms a pattern in the signals that the model overfits on during training. This is observed by its very poor generalization when the same red-green order of label 0 is translated to the lower portion of the image. It led to a complete misclassification and a test set accuracy of 0.

### c. iv) Why is the test accuracy of conv_type=‘replicate’ more than ‘fconv’? 

With replicate the edge pixels are replicated on the padding pixels, unlike reflect, replicate does not change the ordering of the signals, but rather magnify them. This means that the signals of red-left green-right in the label 0 of the training set translates well to the similar signals of the label 0 data in the test set, despite the translation of the signals to the lower portion of the image. Compared to fconv that only pads with 0, which does not help improve the signals after convolutions, replicate leads to a better performance.


### d. i) Run the given code with net_type=‘Net2’, and vary different conv_type from ‘valid’, ‘replicate’, ‘reflect’, ‘circular’, ‘sconv’ and ‘fconv’, and report the validation and test scores in the form of a table.

*******************************************
 Type of convolution :  valid
 Type of network :  Net2
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

*******************************************
 Type of convolution :  replicate
 Type of network :  Net2
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

*******************************************
 Type of convolution :  reflect
 Type of network :  Net2
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

*******************************************
 Type of convolution :  circular
 Type of network :  Net2
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

*******************************************
 Type of convolution :  sconv
 Type of network :  Net2
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

*******************************************
 Type of convolution :  fconv
 Type of network :  Net2
*******************************************
Results for validation dataset {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0, 9: 100.0}
mean: 100.0000 std: 0.0000 for validation
Results for test dataset {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
mean: 0.0000 std: 0.0000 for test

| convolution type | validation accuracy (mean) | validation accuracy (std) | test accuracy (mean) | test accuracy (std) |
| ---------------- | -------------------------- | ------------------------- | -------------------- | ------------------- |
| valid            | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |
| replicate        | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |
| reflect          | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |
| circular         | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |
| sconv            | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |
| fconv            | 100.0000                   | 0.0000                    | 0.0000               | 0.0000              |

### d. ii) Do the test accuracies for each of the conv_types in net_type=‘Net2’ increase or decrease w.r.t their corresponding conv_type counterparts in ‘Net1’?

 The test accuracies for all of the different conv_types decrease in Net2 when compared to Net1. In fact, all test accuracies for Net2 are 0, and their respective validation accuracies  are 100.

### d. iii)  State the reason behind this change in test accuracies

In Net2 after the convolution layers, instead of applying adaptive max pooling which extracts the strongest signal per channel, a flattening operation flattens the tensor to a single dimension. For example, with "conv_types=fconv", the dimensions after the convolution layers is $[64, 6, 6]$, as previously discussed in (c. ii), but the flattening operation flattens the tensor to give a dimension of $[2304]$. By doing this, Net2 has destroyed the spatial information extracted by the convolutions, which as mentioned in the previous sub-questions, is crucial for generalizing on the test dataset. More specifically, instead of extracting the signals regardless of location on the image like a max pooling layer would, the flattening operation restructures the data to a single dimension. Because the signals occur at different regions of the image for the train and test sets, the trained model cannot generalize to the test data where the signals of the red and green rectangles occur in a different section of the flattened data.

## Question 1.2

### a)  Plot the accuracy of the model during inference respect to the angle of rotation of test images and include the figure in your answers. What can you conclude about rotational invariance of CNNs from the plot? Explain why are there larger accuracy at certain degrees.

> [!figure] ![[rotation_acc.png| 400]] 
> *Figure 1*: Accuracy of model during inference respect to angle of rotation of test images. Model not trained on rotated images.

This experiment shows that the trained CNN model could only classify objects in the image (with high accuracy) that are in their original orientation, either not rotated or rotated 360 degrees. This implies that CNNs are not rotationally invariant, because the performance of the trained classifier varied for images rotated for different angles. When the images are rotated to a different angle, the trained model could no longer classify the objects in the image. Since the model is trained on non-rotated images, it could only classify objects that are not rotated, hence the higher accuracies closer to 0 or 360 degrees of rotation.
### b) Train a new model like the previous one but adding rotational augmentation to its datasets, and plot its accuracy during inference with respect to the angle of rotation of the test images. Include the figure in your answers. Describe the differences observed compared to the plot in Question 1.2(a) and explain the reasons for these differences.

> [!figure] ![[rotation_acc_trained.png| 400]] 
> *Figure 2*: Accuracy of model during inference respect to angle of rotation of test images. Model trained with random rotation transformation of up to 360 degrees.

By retraining the model with rotation augmented images, the CNN model is capable of performing better on rotated images at test time. We see an improved test time accuracy for most angles between 0 and 360. This shows that by providing the CNN model with images that are rotated during training, the model has learned to classify rotated objects, in addition to their non-rotated versions. However, we also observe that the accuracies at the extreme angles decreased, from 0.7 down to 0.48. This shows that by introducing rotation to the training images, although the model can generalize better to rotated images at test time, the model has also decreased in performance since the classification task became harder when the images are rotated.
## Question 2.1
### a) Discuss the computational challenges that arise in Transformer models when handling long input sequences. Your response should: (a) Briefly describe why long sequences are challenging for Transformers. (b) Suggest a method to address this challenge.

Transformers rely on the attention mechanism in encoding the pairwise interactions between all tokens of the input sequence. Capturing this pairwise interaction in the input sequence leads to an exponential computational complexity $O(n^2)$. More specifically, the matrix multiplication for calculating the Query-Key similarity ($QK^\top$) that measures how much each token should attend to others tokens produces a $n \times n$ matrix, and for each element in the matrix, the dot product involves $d_{k}$ multiplications and additions. The total number of operations for computing $QK^{\top}$ which has shapes $n\times d_{k}$ and $d_{k}\times n$ respectively, is thus $O(n^2d_{k})$. As $n$ grows to a larger sequence length, the computation grows quadratically, which becomes computationally intensive to calculate.

A method for addressing the quadratic growing computation complexity of the attention mechanism is the use of sparse attention. Methods like local attention limits the attention of a token to only a fixed window size of $w$ where $w \ll n$, each token attends to only $w$ surrounding tokens, and the computational complexity reduces to $O(nw)$. 
### b) Discuss the receptive field of Transformer and Convolutional Neural Networks (CNNs), and their capability to capture long-range dependencies.

Receptive field refers to the region of the input data that a particular neuron or feature in the network is sensitive to, which means the receptive field determines that amount of contextual information from the input a neuron can utilize. 

Convolutional Neural Network have a local receptive field that is determined by the size of the convolutional kernels as each neuron of a convolutional layer is connected to a local region of its input through the kernels. This means that CNNs struggle with capturing long range dependancies, since usual kernel of convolutions are restricted to a small size of $3\times {3}$ or $5\times 5$. Although the receptive field of CNNs grows with the number of layers, each layer of the network only adds a small fixed amount to the total receptive field due to the usually small kernel size and small stride size.

Transformers are capable of capturing long range dependencies as the attention mechanism models a global receptive field where each token attends to every other tokens in the input sequence. In other words, each feature of the network is sensitive to, and depends on, all other regions of the input sequence. This allows transformers to utilize all contextual information in the input sequence.

### c) Explain why the scaling factor $\sqrt{d_{k}}$ is used in the self-attention mechanism (refer to Eq. (4)). Describe its effect on the computation of attention scores.

Without scaling by $\sqrt{ d_{k} }$ in the equation $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{ d_{k} }}\right) \mathbf{V}$, the variance of the dot product in the nominator grows with dimensionality of the query and key vectors $d_{k}$, which leads to larger values as $d_{k}$ increases. This scaling prevents too large values from entering the softmax, which can lead to peaked softmax outputs that destabilize the training through vanishing gradients or can prevent a more balanced attention weights that attends to a range of tokens (due to the peaked output). 

We can show that scaling by $\sqrt{ d_{k} }$ is effective by first assuming that components of the vectors $Q_{i}$ and $K_{j}$ are independent and identically distributed random variables. This means $\mathbb{E}[Q_{il}] = \mathbb{E}[K_{jl}] = 0$ and $Var(Q_{il}) = Var(K_{jl}) = \sigma^2$ for all $l$ in $[1:d_{k}]$. Suppose the 

### d) Explain the advantages of using multiple attention heads compared to a single attention mechanism with the same total computational cost.

