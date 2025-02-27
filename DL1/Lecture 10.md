### **Generative Learning**

There are three eras of generative learning: variational, adversarial, and diffusion. **Generative learning aims to learn the distribution of data itself and then use this knowledge to generate new, unseen samples.** This contrasts with discriminative learning, which focuses on predicting a label given an input.

### Generative vs. Discriminative Learning

- **Discriminative Learning:** Aims to model the conditional probability of a target variable _y_ given input features _x_, denoted as _p(y|x)_. Examples include classifiers that distinguish between different classes.
- **Generative Learning:** Focuses on learning the probability distribution of the data itself, denoted as _p(x)_. This involves understanding the underlying structure and patterns in the data.

**Key Differences:**
- **Normalization:** Discriminative models normalize probabilities for outputs but not inputs. Generative models normalize probabilities for inputs, ensuring that the generated samples are plausible.
- **Applications:** Generative models have broader applications, including generating new samples, data interpretation, and enhancing discriminative models.

**Importance of Generative Learning**
Generative learning offers several key benefits:
- Modeling Data Distribution: It enables us to understand and model the distribution of data itself, leading to insights into the underlying processes generating the data.
- Interpretability: Generative models provide insights into the data, making the models and the data more interpretable.
- Sample Generation: They can generate new samples, creating data that resembles the training data.
- Enhancing Discriminative Models: Generative models can be used to improve the performance of discriminative models by providing prior knowledge about the data distribution.

### **Generative Learning 1: The Variational Era**
#### **Autoencoder**
![[autoencoder.png | 400]]

**The autoencoder is a feedforward network with a bottleneck layer**. The bottleneck layer forces the network to learn a compressed representation of the input data. It consists of two parts:

- **Encoder**: Maps the input to the bottleneck layer, learning a compressed representation.
- **Decoder**: Maps the bottleneck representation back to the output, attempting to reconstruct the original input.

**The architecture and choice of layers for the encoder and decoder are flexible**. Training is straightforward and unsupervised - minimize the error between the input and the reconstructed output. This "auto" part of the name stems from the fact that no labels are needed for training.

**Autoencoders can be used for**:

- Learning lower-dimensional feature representations.
- Compression.
- Invariance.
- Redundancy removal.

Autoencoders can theoretically generate new samples by choosing points in the latent space. However, this is not practical because the **latent space is not constrained, and there is no clear method for sampling from it**.

#### **Variational Autoencoder (VAE)**

**VAEs address the limitations of autoencoders by constraining the latent space to follow a Gaussian distribution**. The encoder and decoder are trained with this Gaussian constraint. During inference, sampling from this constrained space generates new samples.

**Let's break down the probability density functions in a VAE:**

- **Prior distribution** over the latent variable _z_, $p(z)$ is defined by the user, typically a standard normal distribution.
-  **$p_\theta(x|z)$:** The **likelihood** of the data _x_ given the latent variable _z_. This is modeled by the decoder and is parameterized by $\theta$. This distribution is typically chosen to be a Gaussian distribution whose mean is the output of the decoder.
- **Marginal likelihood (evidence):** $p_\theta (x) = \int p_\theta (x|z)p(z)dz$ is the probability of observing data point _x_ under the model. This integral is intractable because it involves integrating over all possible values of the latent variable _z_.

![[autoencoders.png | 400]]

**The posterior distribution** $p_\theta(z|x)$ describes the probability of latent variable z given observed data x. This is the distribution we want to sample from to generate new data. However, **it is intractable to compute directly**.

**We can attempt to approximate the posterior with a neural network** $q_\theta (z|x)$, but this doesn't solve the problem because we still don't know the true posterior $p_\theta (z|x)$.
$$
q_{\phi}^* = \underset{q_{\phi}}{\text{argmin}} \ KL(q_{\phi}(z|x) || p_{\theta}(z|x))
$$
- **Challenge:** The true posterior distribution $p_\theta(z|x)$ is unknown and intractable to compute directly.
- **Approximation:** We aim to approximate the true posterior $p_\theta(z|x)$ with a simpler, tractable distribution $q_\theta(z|x)$.
- **Choice of Approximation:** We typically choose $q_\theta(z|x)$ to be a Gaussian distribution, parameterized by a neural network (the encoder).
#### **ELBO**
The **Evidence Lower BOund (ELBO)** provides a solution by offering a **tractable lower bound on the marginal log-likelihood**. 

**Relationship to KL Divergence:** The ELBO is a lower bound to the marginal log-likelihood, and maximizing the ELBO minimizes the KL divergence between the approximate posterior $q_\theta(z|x)$ and the true posterior $p_\theta(z|x)$.
$$
ELBO(q) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))
$$
**The ELBO consists of two parts**:

- **Reconstruction loss:**  $\mathbb{E}_{q(z|x)}[\log p(x|z)]$. Measures how well the decoder reconstructs the input data from the latent representation. This term encourages the decoder to reconstruct the input data accurately.
- **Regularization loss:** Measures the difference between the approximate posterior distribution $q_\theta (z|x)$ and the prior distribution $p(z)$. This term encourages the latent representation to follow the chosen prior, typically a standard normal distribution.

#### VAEs Summarized
- **Extension of Autoencoders:** VAEs extend autoencoders by introducing a Gaussian constraint on the latent space.
- **Approximation of Posterior:** The true posterior is approximated using a simpler, tractable distribution (Gaussian).
- **ELBO Optimization:** Training involves maximizing the ELBO as a surrogate objective function.
- **Backpropagation Through Random Variables:** Optimizing VAEs requires backpropagating through the random sampling step, which is not directly differentiable.
	- **Reparameterization Trick:** The reparameterization trick addresses this issue by factoring out the Gaussian sampling step from the main computation graph, making the gradients well-defined.
- **Further Reading:** The slide recommends a resource by Yuge Shi, "How I learned to stop worrying and write ELBO (and its gradients) in a billion ways"

### Generative Learning 2: The Adversarial Era

#### **Explicit vs Implicit Density**

- **Explicit density:** Directly model the true data distribution $p^*(x)$ as $p_\theta (x)$.
- **Implicit density:** Sample from $p_\theta (x)$ without defining the distribution explicitly.

GAN
- **Difficulties with Explicit Densities:** Learning explicit densities can be computationally challenging and may require strong assumptions about the data distribution.
- **Focus on Appearance:** In practice, we often care about how generated samples look rather than having a precise mathematical formulation of the density function.
- **GAN Approach:** GANs leverage the idea of implicit density learning to generate high-quality samples.

GAN: High-Quality Generation Through Game Theory

- **Reference:** This slide cites a NeurIPS 2016 tutorial on Generative Adversarial Networks.

#### **What is a GAN?**
- **Generative:** GANs can generate novel, realistic samples from the data distribution.
- **Adversarial:** They involve training two competing models:
    - **Generator:** Creates synthetic samples.
    - **Discriminator:** Evaluates the authenticity of samples, distinguishing between real and fake.
- **Network:** GANs are implemented using deep neural networks and trained using backpropagation.

#### Intuition Behind GANs
- **Analogy:** The training process is analogous to a game between a counterfeiter and a police officer:
    - **Counterfeiter (Generator):** Aims to create fake money that is indistinguishable from real money.
    - **Police (Discriminator):** Tries to detect fake money as accurately as possible.
- **Evolution:** Over time, both the generator and discriminator improve their strategies, pushing each other to become better.
- **Convergence:** The process ideally converges to a Nash equilibrium, where neither model can significantly outperform the other.

#### GAN Architecture

**GANs consist of two neural networks**:
- **Generator network:** $G(z; \theta_G)$, maps a random latent variable _z_ to a generated data sample _x_. It is a differentiable neural network with no invertibility requirement, allowing for flexible modeling.
    - The input _z_ is drawn from a chosen distribution, typically a standard normal distribution or a uniform distribution.
- **Discriminator network:** $D(x; \theta_D)$, receives either real or generated data as input and outputs a probability indicating whether the input is real (1) or fake (0).
	- **Differentiable Network:** Similar to the generator, the discriminator can be any differentiable neural network.
	- **Input:** The discriminator receives either real images from the training set or generated images from the generator.

**Discriminator Loss:**
The discriminator loss is based on binary cross-entropy and is designed to **maximize the likelihood of correctly classifying real and fake data**.
$$
\begin{equation}
\begin{aligned}
J_D(\theta_D, \theta_G) &= \frac{1}{2}\text{BCE}(Data, 1) + \frac{1}{2} \text{BCE}(fake, 0)\\
&= -\frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \frac{1}{2}\mathbb{E}_{x \sim p_{gen}}[\log (1 - D(x))]
\end{aligned}
\end{equation}
$$

![[example_gan.png]]

#### How to Train a GAN
Training GANs is challenging because **there is no ground truth "equivalent" image in the training batch for each generated image**. The generator produces random images independent of the comparison batch.

#### **The Minimax Loss**
To address this challenge, GANs utilize the minimax loss, which frames the training process as a **zero-sum game**.
- **Generator loss:** The generator aims to minimize its loss, which is the negative of the discriminator loss.
- **Discriminator loss:** The discriminator aims to maximize its loss, which corresponds to correctly classifying real and fake data.

The **learning objective** becomes a min-max problem:
$$
min_G max_D V(D,G) = E_{x \sim p_{data}}[logD(x)] + E_{z \sim p_{gen}}[log(1-D(G(z)))]
$$
where:
- $D(x) = 1$ indicates the discriminator believes _x_ is real.
- $D(G(z)) = 1$ indicates the discriminator believes the generated data _G(z)_ is real.


#### Heuristic Non-Saturating Loss
In practice, a heuristic non-saturating loss is often used to improve training stability. This loss **encourages the generator to learn even when the discriminator is highly accurate on real images**. It achieves this by modifying the generator's objective to focus on maximizing the probability of fooling the discriminator.
- **Discriminator Loss:** Remains similar to the previous formulation, maximizing the likelihood of correctly labeling real and fake data.
- **Generator Loss:** Instead of minimizing the negative discriminator loss, the generator aims to maximize the likelihood of the discriminator being wrong.

The non-saturating losses are defined as follows:

- **Discriminator loss:** $J_D = -\frac{1}{2}E_{x \sim p_{data}}[logD(x)] - \frac{1}{2}E_{z \sim p_z}[log(1-D(G(z)))]$ (same as minimax loss)
- **Generator loss:** $J_G = -\frac{1}{2}E_{z \sim p_z}[log(D(G(z)))]$


#### **Challenges in Training GANs**

Training GANs is notoriously difficult due to several factors:

1. **Vanishing gradients**: If the discriminator becomes too good at distinguishing real from fake data early in training, the gradients for the generator vanish, hindering learning. This happens because the generator struggles to receive meaningful feedback when the discriminator is already near perfect.
    - **Weak Discriminator:** A bad discriminator provides confusing gradients to the generator.
    - **Near-Perfect Discriminator:** A very accurate discriminator produces gradients close to zero, leading to slow or no learning for the generator.
2. **Batch normalization**: Batch normalization can introduce unwanted correlations within mini-batches, leading to generated images that appear smooth but lack diversity. Removing batch normalization is a common solution.
3. **Convergence**: GAN training involves a complex optimization process that can be unstable. Finding a saddle point does not guarantee a global minimum and is susceptible to disturbances. Reaching an equilibrium point, where neither the generator nor the discriminator can significantly improve, may take an extended training time (weeks). **Mode collapse**, a severe form of non-convergence, occurs when the generator produces only a limited variety of samples, collapsing the diversity of the generated data distribution.
4. **Mode collapse**: Mode collapse happens when the generator focuses on producing samples from a single mode of the data distribution, ignoring other modes and resulting in limited sample diversity. A straightforward technique to mitigate this is to **introduce a penalty for high similarities within a mini-batch**. This encourages the generator to explore a wider range of data modes and generate more diverse samples.
#### Evaluating GANs

- **Evaluation Challenges:** Evaluating generative models, including GANs, is generally difficult.
- **Image Quality Measures:** Metrics like the Fréchet Inception Distance (FID) can assess image quality, but they have limitations.
- **Human Evaluation:** The most reliable evaluation often involves human judgment, where people rank or assess the quality and diversity of generated samples.

### **Generative Learning 3: The Diffusion Era**

#### **Back to Autoencoders and Gaussians**
- **VAE Limitations:** VAEs attempt to directly map input data to a Gaussian latent space, which can be challenging.
- **Gradual Approach:** Diffusion models offer a more gradual approach to transforming data into a Gaussian distribution.

#### **Denoising Diffusion Models**
**Diffusion models use a two-step process**:
1. **Forward process (encoding)**: Gradually adds noise to the input data until it resembles a Gaussian distribution. This process is guided by a variance schedule that determines how much noise to add at each step.
	- **Scaling and Noise Addition:** Each step in the forward process typically involves scaling down the input and adding Gaussian noise.
	- **Variance Schedule:** A variance schedule controls the amount of noise added at each step.
2. **Reverse process (decoding)**: Learns to reverse the noise-adding process and generate data samples from the Gaussian noise.

#### **Training Diffusion Models**
During training, the model learns to predict the slightly less noisy version of the image given a noisy input. By iteratively applying this denoising process, the model can eventually generate clean images from pure Gaussian noise.
#### **Sampling from Diffusion Models**
To sample from a trained diffusion model, you start with Gaussian noise and iteratively apply the learned denoising process. At each step, the model refines the noisy image, gradually removing the noise and generating a realistic sample.

#### Impact of Diffusion Models
- **State-of-the-Art Performance:** Diffusion models have driven significant advances in generative models, powering tools like Midjourney, Imagen, Sora, DALL-E, and Stable Diffusion.
- **Intuitive Process:** They operate directly in pixel space, making the generation process more intuitive.
- **Memory Considerations:** Working directly in pixel space can lead to memory constraints, especially for high-resolution images.
- **Extensions:** Latent diffusion models and conditional diffusion models are examples of extensions that address some of the limitations.

#### Ethical Side of Generative Learning

- **Blurred Reality:** Generative models blur the line between real and generated data, raising ethical concerns.
- **Intellectual Property:** Training data for large models often involves copyrighted material, posing questions about intellectual property rights.
- **Societal Impact:** Concerns exist regarding deepfakes, potential for manipulation, and the impact on job markets.
- **Real-World Implications:** The widespread use of generative models necessitates careful consideration of their ethical and societal implications.

### Model Summary

![[generative_models.png]]