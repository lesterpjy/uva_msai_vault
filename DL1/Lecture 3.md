## Deep Learning Optimization


Optimization vs. Learning
Learning suggests we want to generalize on unobserved data, optimization just means to minimize certain objective function given some observed data.

Empirical Risk Minimization
In practice we only have an empirical data distribution $\hat{p}_{data}$, defined by a set of training examples. We train to minimize
![[emperical_risk.png | 400]]
To minimize, we take a negative gradient descent.
![[negative_step.png | 180]]

Why does mini-batch SGD work?
- Training sample may have lots of noise or outliers or biases, a randomly sampled minibatch may reflect the true data generation distribution better/worse.
- Full gradient may get stuck in local minima, while minibatch gradient may not.

| Gradient Descent                                          | Stochastic Gradient Descent                            |
| --------------------------------------------------------- | ------------------------------------------------------ |
| Computes gradient using the whole Training dataset        | Computes gradient using a single training sample       |
| Slow and computationally expensive algorithm              | Faster and less computationally expensive than GD      |
| Deterministic                                             | Stochastic                                             |
| Gives optimal solution given sufficient time to converge* | Gives good solution but not optimal                    |
| No random shuffling of points are required                | Shuffling needed. More hyperparameters like batchsize. |
| Can’t escape shallow local minima easily                  | SGD can escape shallow local minima more easily        |
| Convergence is slow                                       | Convergence much faster                                |

**Challenges in Optimization**
Neural network training is **non-convex** optimization: it involves a function which has multiple optima, and is extremely difficult to locate the global optimum.
1. Ill conditioning → a strong gradient might not even be good enough
2. Local optimization is susceptive to local minima
3. Ravines, plateaus, cliffs, and pathological curvatures
4. Vanishing and exploding gradients
5. Long-term dependencies

### Ill Conditioning
Hessian matrix $H$: square matrix of second-order partial derivatives of a scalar-valued function. Hessian describes the local curvature of a function of many variables.
Critical points – Hessian matrix
- A local minimum: positive definite (all its eigenvalues are positive)
- A local maximum: negative definite (all its eigenvalues are negative)
- A saddle point: at least one eigenvalue is positive and at least one eigenvalue is negative. Why is this bad?

Saddle points are problematic in optimization because:
- **Flat Directions Slow Down Optimization**: At a saddle point, the gradient in some directions is nearly zero or close to zero, leading to very slow progress when using gradient-based methods.
- **Ambiguity of Direction**: In some directions, the saddle point behaves like a local minimum (positive curvature), and in others, it behaves like a local maximum (negative curvature). This ambiguity can confuse optimization algorithms, especially if second-order information like the Hessian is not explicitly utilized.
- **Prevalence in High Dimensions**: In high-dimensional spaces, saddle points are exponentially more common than local minima or maxima. This means optimizers encounter saddle points frequently, hindering efficient convergence.
- **Misleading Dynamics**: Near a saddle point, the gradient may still point toward the "center" of the saddle, causing optimization algorithms to oscillate or get stuck in loops rather than making consistent progress toward a better solution.
- **Impact on Loss Surface Exploration**: Saddle points can trap optimization algorithms in regions of the loss surface that are suboptimal, delaying progress toward finding valleys of lower loss.
When saddle points dominate, many eigenvalues of the Hessian may be close to zero, leading to numerical instabilities and further challenges in optimization.

Condition number: ratio of the magnitude of the largest (i) and smallest eigenvalue (j) of the Hessian. With a poor (large) condition number, gradient descent performs poorly.
- In one direction derivative increases rapidly, in another it increases slowly.
- also makes it difficult to choose a good step size

### Local Mininma
Local optimization is susceptive to local minima
- Model identifiability: A model is said to be identifiable if a sufficiently large training set can rule out all but one setting of the model’s parameters. 
- Models with latent variables are often **not identifiable** because we can obtain equivalent models by exchanging latent variables. Ex: In a neural network, swapping two neurons in the same hidden layer does not affect the output or cost function, creating multiple equivalent parameter configurations that results in the same cost.
- **Types of local minima**:
	- **Benign local minima**: These result from non-identifiability (e.g., symmetry in parameter configurations) and are equivalent in cost value. They do not affect the model's performance because all such configurations are equally good solutions.
	- **Problematic local minima**: These are true suboptimal points on the loss surface, where the cost is higher than the global minimum. These are more concerning because they can prevent the model from achieving optimal performance.

### Ravines
![[ravines.png]]

### Plateaus and flat areas
Near zero gradients in flat areas, hence no, or very slow learning.

But why are flat minima still preferred? → Better generalization
![[flat_minima_generalization.png]]

### Cliffs and exploding gradients
Neural networks with many layers often have steep regions resembling cliffs.
These result from the multiplication of several large weights together.

Simple trick to deal with exploding gradients: when $||g||>\eta$, the gradient vector is scaled down such that its norm becomes exactly $\eta$
$$
g \leftarrow \frac{\eta g}{||g||} \ \ \ \text{if} \ |g|>\eta
$$
combining flat areas with very steep minima → very challenging
![[flat_steep_minima.png]]

### Long-term dependencies
Especially for networks with many layers or recurrent neural networks.
The vanishing and exploding gradient problem
- Certain functions lead to a scaling of the gradient (potentially often).
- Vanishing gradients -> no direction to move
- Exploding gradients -> learning unstable.
For training-trajectory dependency: hard to recover from a bad start!


## Advanced Optimizers
Gradient descent can be enhanced.
![[improving_gradient_descent.png | 400]]

### Momentum
Momentum can be understood through the lens of **exponentially weighted moving averages** (EWMA). This involves tracking the historical gradients and combining them with the current gradient to create a smoother update trajectory.
$$
V_{t}​=\beta V_{t-1} + (1−\beta)S_{t}
$$
$\beta$: The **momentum coefficient** (or smoothing factor), controlling how much past values contribute to the current update. At the beginning of training, $V_{t}$ is biased toward the initial value $V_{0}=0$ leading to underestimated momentum. Bias correction:
$$
\hat{V}_{t} = \frac{V_{t}}{1-\beta^t}
$$
where $t$ is the iteration.
**Momentum main idea**: Don’t switch update direction all the time.
Maintain "momentum" from previous updates → dampens oscillations.
$$
\begin{equation}
\begin{aligned}
v_{t+1} &= \gamma v_{t} + \eta_{t}g_{t} \\
w_{t+1} &= w_{t} - v_{t+1}
\end{aligned}
\end{equation}
$$
Exponential averaging keeps steady direction.

### Nesterov Momentum
Use future gradient instead of current gradient:
$$
\begin{equation}
\begin{aligned}
v_{t+1} &= \gamma v_{t} + \eta_{t} \nabla_{w}(w_{t}-\gamma v_{t}) \\
w_{t+1} &= w_{t} -v_{t+1}
\end{aligned}
\end{equation}
$$
Prevents going too fast and increase responsiveness.

### Adaptive Step Sizes
Adaptive Gradient Algorithm – **Adagrad**:
$$
\begin{equation}
\begin{aligned}
w_{t+1} &= w_{t} - \frac{\eta}{\sqrt{ r }+\epsilon} \odot g_{t} \\
\text{where} \ \ r &= \sum_{t}(\nabla_{w}\mathcal{L})^2
\end{aligned}
\end{equation}
$$
- The learning rate is adapted component-wise to the parameters by incorporating knowledge of past observations.
- Rapid decrease in learning rates for parameters with large partial derivatives.
- Smaller decrease in learning rates for parameters with small partial derivatives.

**Adadelta**: Adagrad++
- Seeks to reduce its aggressive, monotonically decreasing learning rate.
- Restricts the window of accumulated past gradients to some fixed size, instead of accumulating all past squared gradients.
- No need to set a default learning rate, as it has been eliminated from the update rule.

**RMSprop**
$$
\begin{equation}
\begin{aligned}
r_{t} &= \alpha r_{t-1} + (1-\alpha)g_{t}^2 \\
v_{t} &= \frac{\eta}{\sqrt{ r_{t} }+\epsilon} \odot g_{t} \\
w_{t+1} &= w_{t} - v_{t}
\end{aligned}
\end{equation}
$$
Large gradients, e.g., too “noisy” loss surface
- Updates are tamed
Small gradients, e.g., stuck in plateau of loss surface
- Updates become more aggressive

**Adam**
![[adam.png | 400]]

Combines RMSprop and momentum.
- Computes adaptive learning rate for each parameter.
- Keeps an exponentially decaying average of past gradients (momentum).
- Introduces bias corrections to the estimates of moments.
Adam is in general better for complex models, but might fail on easy/stupid tasks compared to simple methods like SGD

## Network Initialization
![[weight_init.png]]

Zero Initialization
• Leads to symmetric hidden layers.
• Makes your network no better than a linear model.
• Setting biases to 0 will not create any problems.
Random Initialization
• Breaks symmetry.
• Prevents neurons from learning the same features

- There are two forces on the weight magnitude: small weights are needed to keep data around origin, but large weights are required to have strong learning signals
- Initialization should preserve variance of activations (input variance ≈ output variance to keep distribution between modules same)
- Depends on non-linearity and data normalization
- To maintain data variance, the variance of the weights must be $\frac{1}{d}$ where $d$ is number of input neurons $\Rightarrow$ sample weight values from $w \sim \mathcal{N}\left( 0, \frac{1}{d} \right)$
- **Kaiming initialization**: ReLU set half of the output neurons to 0 $\Rightarrow$ double the weight variance to compensate zero flat-area: $w \sim \mathcal{N}\left( 0, \frac{2}{d} \right)$
- Xavier initialization (assumes linear activation, approx for tanh): initialize weights from $U\left[-\sqrt{ \frac{6}{d_{l-1}+d_{l}} }, \sqrt{ \frac{6}{d_{l-1}+d_{l}} }\right]$, where $d_{l-1}$ is no. of input, $d_{l}$ is no. of output.
- For sigmoid, $U\left[-4\cdot\sqrt{ \frac{6}{d_{l-1}+d_{l}} }, 4\cdot\sqrt{ \frac{6}{d_{l-1}+d_{l}} }\right]$.

