This lecture, "**The Oddities of Deep Learning,**" covers various unconventional aspects and challenges within the field of deep learning.

### Catastrophic Forgetting and Continual Learning

- **Catastrophic Forgetting:** Deep learning models often struggle to retain knowledge from previously learned tasks when trained on new data. This phenomenon, known as catastrophic forgetting, poses a significant obstacle to developing truly intelligent systems capable of continuous learning.
- **Continual Learning:** The goal of continual learning is to enable models to learn sequentially from new data without losing previously acquired knowledge. This involves strategies to mitigate catastrophic forgetting and promote knowledge retention.

### Addressing Catastrophic Forgetting

- **Stability-Plasticity Trade-off:** Achieving a balance between stability (preserving existing knowledge) and plasticity (adapting to new information) is crucial for effective continual learning.
- **Experience Replay:** One approach to combat forgetting is experience replay, where a portion of old data is retained and mixed with new data during training. Techniques for selecting samples for replay include random selection, choosing the most prototypical examples, or using the highest-scoring samples.
- **Challenges of Experience Replay:** Scaling experience replay to a large number of classes or continuous domains, as encountered in large language models (LLMs), remains a challenge.
- **Real-world Data Streams vs. Benchmarks:** Current continual learning benchmarks often simplify the complexities of real-world data streams, which exhibit gradual and sharp drifts, new domains and classes emerging over time, repetition of old information, imbalanced distributions, and temporal consistency as a valuable signal.

### Adversarial Attacks

- **Vulnerability of Deep Learning Models:** Deep learning models can be surprisingly susceptible to adversarial attacks, where carefully crafted input perturbations lead to misclassifications.
- **Formulating Adversarial Attacks:** Adversarial attacks aim to find an input example (x*) that maximizes the model's prediction error while remaining close to the original input (x).
    - **Non-Targeted Attacks:** These attacks seek any incorrect prediction. $$ max_{x*} ; d(f(x^*), y), ; s.t. ; d(x, x^*) < \epsilon $$
    - **Targeted Attacks:** These attacks aim to mislead the model towards a specific target prediction (y*). $$ max_{x*} ; d(f(x^*), y^*), ; s.t. ; d(x, x^*) < \epsilon $$
- **Fast Gradient Sign Method (FGSM):** A basic white-box attack method that uses the gradient of the loss function to determine the direction of input perturbation to maximize error. $$ x^* = x + \epsilon ; sign(\nabla_x L(f(x), y)) $$
- Projected Gradient Descent (PGD): iteratively follows the gradient w.r.t. the data, but clips change at each timestep to make sure the perturbed image stays "close" to the original. This is done by iteratively taking a FGSM step with step size α, and then keeping the perturbed image within the ball of size ϵ around the original image.
- **White-Box vs. Black-Box Attacks:**
    - **White-box attacks** require access to model parameters (like FGSM).
    - **Black-box attacks** operate without knowledge of model internals and often rely on techniques like random noise or gradient walks.
- **Defense Against Adversarial Attacks:** A fundamental solution lies in developing models that are inherently more robust and less prone to drastic changes in predictions due to minor input variations.

### Long-Tailed Deep Learning

- **Real-World Data Imbalance:** Real-world datasets often exhibit long-tailed distributions, where a few classes dominate, while many others have limited samples. This imbalance poses challenges for model training, as models tend to overfit to the majority classes.
- **Addressing Long-Tail Distributions:**
    - **Subsampling:** Reducing the number of samples from common classes.
    - **Oversampling/Resampling:** Increasing the representation of rare classes.
    - **Augmentation:** Applying data augmentation techniques more frequently to rare classes.
    - **Cost-Sensitive Learning:** Scaling the loss function based on class frequency to penalize misclassifications of rare classes more heavily.
    - **Fixed Logit Adjustments:** Applying adjustments to the model's output logits to account for class imbalance.

### Deep Learning on One Sample

- **Deep Image Prior:** The concept that the structure of convolutional neural networks themselves captures a significant amount of image statistics, even without training on large datasets.
- **Single-Image Training:** Training a network as an autoencoder on a single image can reveal surprising capabilities, such as denoising, super-resolution, and inpainting.

### Other Oddities

- **Knowledge Distillation:** Training smaller networks to mimic the predictions of larger, more complex models to reduce model size and complexity.
- **Label Noise:** Introducing a small amount of label noise (intentional mislabeling) during training can sometimes improve generalization.
- **Neural Fields / Implicit Representations:** Representing data using neural networks as a continuous function, allowing for novel ways to manipulate and generate data.