
## Convolutional Neural Networks

**Structure of latent representations**
![[latent_manifold.png]]


### Classification Architectures
#### VGGNet
- All filter sizes are $3\times 3$, as this is the smallest filter size, and is more parameter efficient to build up large filters (receptive field increases over layers).
- Additional non-linearity between filters (more non-linearities for the same "size" of patterns learned).
- $1 \times 1$ convolutions used to increase non-linearity/complexity without increasing receptive field.
![[vggnet.png | 300]]

#### Inception (GoogLeNet)
- Receptive fields should vary in size as objects can appear in different scales
- Naively stacking more convolutional operations on top of each other is expensive and prone to overfitting
- Inception module applies different filter sizes on same input ($1 \times 1$ convolutions for feature reduction)
- Architecture consists of 9 Inception blocks
- Solution for vanishing gradients: have intermediate classifiers that amplify the gradient signal for early layers (loss at these classifiers backprop through the early layers) Only used during training, last classifier used for inference.
- InceptionV2: $5 \times 5$ replaced by two $3 \times 3$ filters
- InceptionV3: Factorize $n \times n$ filters into $1 \times n$ and $n \times 1$ filters (cheaper to compute)
- BatchNormalization has shown to be very helpful in this architecture
![[inception.png]]

#### ResNet
Deeper networks are harder to optimize, and might actually achieve worse results than shallow ones because of depth: adding identity layers increase training error. Performance degradation not caused by overfitting.
![[depth_degradation.png | 350]]

**Residuals**
Say we have the network nonlinearity $a = F(x)$. 
Perhaps easier to learn a function $a = F(x)$ to model differences $a \sim \delta y$ than to model absolute difference $a \sim y$
$\rightarrow$ Let neural networks explicitly model difference mappings
$$
H(x) = F(x) + x
$$
![[residual_block.png | 400]]

Different ways for modeling F (x). Most popular ones shown in figure. BatchNormalization has been shown to be very important because of vanishing gradients.
![[resnet_variations.png | 500]]

Networks with skip connections (like ResNets) converge faster than the same networks without skip connections.

**DenseNets**
Use skip connections to multiple forward layers. Creates complex blocks where last layer sees inputs to all previous layers.

### Tracking/ Detection Architectures

#### R-CNN
- Each box in an image is a region of interest
- Steps: find interesting boxes, rescale to feed through CNN, finally an SVM to see what object the box contains.
- Region proposals extracted with selective search.
- Improve region proposals with linear regression to generate tighter bounding box.
- Cons: each region proposal needs new pass, region proposal determined a priori.
![[rcnn.png | 400]]

#### Fast R-CNN
![[fast_rcnn.png]]
 - Based on middle feature map, get bounding boxes by e.g. selective search
 - RoI pooling returns fixed size feature map for selected bounding box (puts e.g. 3 × 3 mask on features and pools accordingly (stride defined relatively to feature map size))
 - Features used to generate class prediction and bounding box correction (joint training)
 - During training, sample multiple candidate boxes from image and train on all of them. Makes it more efficient/faster, but batch elements might be highly correlated (in the paper, they report that they experienced it to be neglectable).
 - Very accurate and fast, but external box proposals needed.
 - Faster R-CNN: train network to propose box locations (region proposal network), bounding box tightened with regressor.
![[region_proposal_network.png | 300]]

#### Focal Loss
- Cross-entropy is a problem for object detection.
- Many boxes evaluated, a lot of small losses lead to a huge bias.
- Solution: add an exponent to the cross-entropy term!
![[focal_loss.png | 400]]


**Additional topics**
- Mask R-CNN: adding a branch to Faster R-CNN that outputs a binary mask that says whether or not a given pixel is part of an object.
![[mask_rcnn.png | 300]]
- Instead of segmentation within bounding box, just classify pixel directly to get full segmentation mask. (Deconvolution to upscale feature map to image size)
![[pixel_level_classification_deconv.png | 500]]
