## Convolutions
Incorporating prior knowledge (inductive bias) of images into the design.
Reminder:
**2D Convolution**
$$
S(i,j) = (I \circledast K)(i,j) = \sum_{m} \sum_{n} I(i-m,j-n)K(m,n)
$$
**2D Cross-Correlation**
$$
S(i,j) = (I \circledast K)(i,j) = \sum_{m} \sum_{n} I(i+m,j+n)K(m,n)
$$
**Output size of filtering**
$$
\text{output size} = \frac{\text{input size} + 2 \times \text{padding size} - \text{kernel size}}{\text{stride size}} + 1
$$

### Motivations for convolutions
**Sparsity**
- Receptive field is the kernel/filter size.
- Connections are local in space, full in depth
- Receptive field of the units in the deeper layers of a convolutional network is larger than the receptive field of the units in the shallow layers
**Parameter Sharing**
- Learned filter used at every position of the input.
**Equivariance**
- Parameter sharing causes layer to have equivariance to translation.
- A function is equivariant if the input changes, the output changes in the same way.

### Boarder Padding
![[padding.png | 500]]

### Convolution Network
How to retain 3D structure of image data (h, w, channel) in neural networks?
**3D Activations**
![[3d_activations.png | 400]]

Consider a region of the 3D input, $x^r$, with output neuron $h^r$, the output neuron sums over the 3 axis of the input $ijk$:
![[3d_activations_2output.png | 400]]

This can be repeated across the input dimension for $D$ dimensions:
![[3d_activation_weight_sharing.png | 400]]

With weight sharing, this is called **convolution**, without weight sharing, this is called a **locally connected layer**.
![[3d_activations_d_filters.png | 400]]

All together, the weights are 4 dimensional: (output depth, input depth, kernel h, kernel w)

**1x1 convolutions** 
- Convolution operation where the filter size is 1x1, applied across the depth (channels) of the input. Instead of learning spatial features (like larger filters), a 1x1 convolution focuses on learning relationships between channels at a single spatial location.
- Combines information from all input channels to produce new feature representations.
- Can be used as a pre-processing step before applying larger filters to reduce computational cost: reduce no. channels first then subsequent conv.

**Dilated convolutions**
- enlarge receptive field without increasing parameter count (spatial resolution does not change)
- difference between strides and dilation
	- Think of stride as **jumping over parts of the input**, effectively ignoring them. In contrast, dilation is like **spreading out the filter's "reach"** while still applying it densely across the input.
![[dilated_convolutions.png | 450]]

### Pooling
**Max pooling**
![[maxpooling.png]]

Backpropagation for max pooling
- In forward pass, store index that took the max
- Backprop gradient is the input gradient at that index