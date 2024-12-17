### Self-Supervised Learning

Issues with Supervised Learning
- Label Independence Assumption: Assuming labels are independent is often false, as images contain multiple interacting objects.
- Oversimplification: A single label might not capture the complexity of an image.
- Limited Generalization: Models may struggle with unseen scenarios or objects.

- **What is Self-Supervised Learning?**
    - Self-supervised learning aims to train deep networks without labeled data by leveraging the inherent structure of data, such as images and videos.
    - It addresses the bottleneck of data annotation by utilizing the vast amount of unlabeled data available.
        - Reduce reliance on costly human-labeled datasets.
        - Use the structure of data to create "proxy" tasks for pretraining.
    - Two phases of SSL
        - Pretraining: Learn general-purpose representations by solving a **proxy task** (e.g., predicting rotations or relative positions of patches).
        - Downstream Task: Fine-tune representations using sparse labeled data for specific applications.
- **Why Self-Supervised Learning?**
    - **Scalability**: Unlabeled data is abundant, enabling training on larger datasets. For example, ImageNet has around 1 million annotated images, while platforms like Instagram have billions of images.
    - **Generalizability**:
        - Models trained on supervised data often struggle with domain shifts.
        - SSL encourages better adaptability across tasks and data distributions.
    - **Human Learning Analogy**: Humans learn a lot through self-directed exploration and observation, suggesting potential for similar mechanisms in AI.
    - **Label Imperfections**: Labels can be subjective, biased, or erroneous, and self-supervised learning reduces reliance on potentially flawed labels.

How to Train Deep Networks Without Labels
**Self-Supervised Visual Learning**
- Exploits the inherent structure of images and videos to devise pretext tasks for training
- Adapted supervised loss functions to work without labels

### Early Attempts
- Relative Positioning
    - Predict spatial relationships between image patches (e.g., which patch is above another).
    - Inspired by **Word2Vec** in NLP.
    - Model Setup:
        - Split an image into patches.
        - Train a CNN classifier to predict relative positions.
- Colorization
    - Predict the color information of grayscale images.
    - Setup:
        - Input: Grayscale image
        - Output: Color channels
        - Loss: Minimize the difference between predicted and actual colors.
- Rotations
    - Predict the correct rotation of an image (e.g., 0°, 90°, 180°, 270°).
    - Assumption: Understanding an object involves recognizing its natural orientation.

### Modern SSL
**Contrastive self-supervision** is widely used, where augmented versions of the same image are pulled together, while different images are pushed apart in the embedding space. This encourages learning features that are invariant to augmentations and unique to each image.
- Enforce two properties:
    1. Uniqueness: Each image must have a distinct representation.
    2. Augmentation Invariance: Representations must remain consistent across transformations.
- Contrastive Loss:
$$
\mathcal{L}_{ij} = -\log \frac{\exp(\mathbf{sim}(z_{i}, z_{j})/\tau)}{\sum_{k}\exp(\mathbf{sim}(z_{i},z_{k}))/\tau}
$$
	- $\mathbf{sim}$: Cosine similarity between embeddings $z_{i}$ and $z_{j}$
	- $\tau$: Temperature parameter that controls sensitivity.
- Training Process
    - Positive Pair: Different augmentations of the same image.
    - Negative Pair: Different images.
    - Pull positives together and push negatives apart.


Self-Supervised Video Learning
- Videos provide an additional signal of time, which can be leveraged for pretext tasks.
    - Examples:
        - Predicting temporal order of frames
        - Predicting if a video is played in reverse
        - Predicting alignment between audio and video
        - Predicting the pace of a video
        - Identifying the odd-one-out frame

Self-Supervised Learning on Other Modalities
- Examples of applying self-supervised learning on modalities beyond images and video include:
    - Audio
    - Text: success of Large Language Models (LLMs) raises questions about whether they are considered self-supervised

Performance of Self-Supervised Models
- DINOv2, a contrastive learning model trained on a massive dataset, demonstrates the effectiveness of self-supervised learning.
- DINOv2 shows improved robustness compared to supervised networks when deployed in new settings.

### Vision-Language Models
These models aim to bridge the gap between visual and textual information, going beyond simple labels.
- **CLIP (Contrastive Language-Image Pre-training)** is a prominent example.
	- Key Idea: Treat semantics beyond labels by using sentences to describe images.
	- Architecture: Two encoders, one for images and one for text, are trained jointly using contrastive loss to align image-text pairs.
	- Success Factors:
		- Large-scale pre-training on a dataset of 400 million image-text pairs.
		- Sentences provide richer semantic context than labels.
		- Continuous representation of semantics.
	- Zero-Shot Capabilities: CLIP can perform tasks like classification without task-specific training, showcasing its powerful generalization abilities.
	- Robustness: CLIP demonstrates improved performance on datasets with distribution shifts compared to other pre-trained models.

- **Applications of CLIP**:
	- Zero-shot classification
	- Text-guided image generation (e.g., DALLE-2)
	- Video-language retrieval (e.g., CLIP4Clip)
	- Semantic segmentation (e.g., GroupViT)
- **Limitations of CLIP**:
	- Limited to similarity scores between text and images.
	- Struggles with open-ended tasks that require language generation.
	- Relies on a fixed prompt mechanism for tasks like classification.
	- Fine-tuning can sometimes decrease performance.
- **Prompt Engineering**: Carefully designing prompts can significantly influence CLIP's performance.
- **Learning to Prompt**: Techniques like Context Optimization aim to automatically learn optimal prompts

**Flamingo**
- A transformer-based architecture for multimodal few-shot learning tasks like image captioning, visual dialogue, and visual question answering.
- Capable of in-context learning with images and text as context, similar to GPT.
- Architecture:
	- Vision side: Encoder using a contrastive text-image approach like CLIP.
	- Language side: Pre-trained autoregressive language model.
	- Perceiver attention component links vision and language sides, outputting visual tokens to condition the language model.
![[flamingo.png]]
