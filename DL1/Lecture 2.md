## Forward and Backward Propagation

Deep learning: A family of parametric, non-linear and hierarchical **representation learning functions**, which are massively optimized with stochastic gradient descent to encode domain knowledge, ie, domain invariances, stationarity.
### Deep feedforward networks
- Also called multi-layer perceptrons
- Approximates some function $f$ that defines the mapping: $y = f(x; \theta)$
- Learns the value of the parameters $\theta$ with the best function approximation
- No feedback connections: RNN has feedback connections, and brains have feedback connections
Formed with composite functions:
$$
y = a_{L} = f(x;\theta) = h_{L} \ \circ h_{L-1} \ \circ \dots \ \circ h_{1} \ \circ x
$$
where each functions $h_{l}$ is parameterized by parameters $\theta_{l}$.
![[ffn.png]]

Requirements
1. Activations must be 1st-order differentiable (almost) everywhere
2. Take special care when there are cycles in the architecture of blocks: unfold the graph

#### Training goal and overview
Dataset of input and output
Initialize weights and biases with random values
Learn weights and biases through "forward-backward" propagation
- Forward step: map input to predicted output
- Loss step: compare predicted output to ground truth output
- Backward step: correct predictions by propagating gradients
![[training_ffn.png | cetner | 400]]

#### Activation Functions
Why is activation necessary?
Combination of linear functions is a linear function:
$$
\begin{equation}
\begin{aligned}
v(x) &= ax + b \\
w(z) &= cz + d \\
w(v(x))  &= c(ax+ b) + d = (ac)x + (cb + d)
\end{aligned}
\end{equation}
$$
Activations are necessary to obtain non-linear functions.
If output range of activation is limited, then called a "squashing function."
Activation functions need to be differentiable at most points.

##### Sigmoid
- Output range of $(0,1)$ $\Rightarrow$ always positive output with mean of $0.5$
- For output to emulate probabilities
- Saturate at extreme
$$
\begin{equation}
\begin{aligned}
\sigma(z) &= \frac{1}{1+e^{-z}} \\
\frac{d}{dz}\sigma(z) &= \sigma(z)(1-\sigma(z))
\end{aligned}
\end{equation}
$$
![[sigmoid.png | center | 300]]
##### Tanh
- Output range of $[-1, + 1]$ $\Rightarrow$ mean of 0 instead of 0.5, no positive bias
- Symmetric behavior around 0 provides wider range of stronger gradients compared to Sigmoid.
- Saturate at extreme
$$
\begin{equation}
\begin{aligned}
h(x) &= \frac{e^x-e^{-x}}{e^x+e^{-x}} \\
\frac{\partial}{\partial x}h(x) &= 1-\tanh^2(x) \\
\end{aligned}
\end{equation}
$$
![[tanh.png | center | 300]]
##### Rectified Linear Unit (ReLU)
- Sparse activation: in randomly initialized network, ~50% active
- Better gradient propagation: fewer vanishing gradient problems compared to sigmoid that saturates in both directions.
- Efficient computation: composed of only conditional, addition and multiplication.
- Non-differentiable at 0 (in practice derivative at 0 arbitrarily set)
- Not 0 centered
- Unbounded
- Dead neurons problem: neurons pushed into state that are inactive for all inputs.
$$
\begin{equation}
\begin{aligned}
h(x) &= max(0,x) \\
\frac{\partial}{\partial x}h(x) &=
\left\{ \begin{array}{rcl}
1 \ \ \ \text{when}\ \ \  x >0\\
0 \ \ \ \text{when}\ \ \  x \leq 0
\end{array}\right.
\end{aligned}
\end{equation}
$$
![[relu.png | center | 300]]
##### LeakyReLU
- Allows a small positive gradient when the unit is not active.
- Parametric ReLUs (PReLU) treat $a$ as a learnable parameter.
$$
\begin{equation}
\begin{aligned}
h(x) &=  
\left\{ \begin{array}{rcl}
x \ \ \ \text{when}\ \ \  x >0\\
ax \ \ \ \text{when}\ \ \  x \leq 0
\end{array}\right.  \\
\frac{\partial}{\partial x}h(x) &=
\left\{ \begin{array}{rcl}
1 \ \ \ \text{when}\ \ \  x >0\\
a \ \ \ \text{when}\ \ \  x \leq 0
\end{array}\right.
\end{aligned}
\end{equation}
$$
![[leakyrelu.png | center | 300]]
##### Exponential Linear Unit (ELU)
- Smooth approximation to the rectifier
- Non-monotonic "bump" when $x<0$
- Default activation for models like BERT
$$
\begin{equation}
\begin{aligned}
h(x) &=  
\left\{ \begin{array}{rcl}
x \ \ \ \text{when}\ \ \  x >0\\
ax \ \ \ \text{when}\ \ \  x \leq 0
\end{array}\right.  \\
\frac{\partial}{\partial x}h(x) &=
\left\{ \begin{array}{rcl}
1 \ \ \ \text{when}\ \ \  x >0\\
a \ \ \ \text{when}\ \ \  x \leq 0
\end{array}\right.
\end{aligned}
\end{equation}
$$
![[elu.png | center | 300]]

##### How to choose activation?
![[how2choose.png | center |  500]]
#### Cost Functions
##### Binary classification
- binary label of example $i$: $y_{i} \in \{0,1\}$
- Output of example $i$: $p_{i} \in [0,1]$
- Goal minimize $p_{i}$ if $y_{i}=0$, maximize if $y_{i}=1$
- Maximize: $p_{i}^{y_{i}}(1-p_{i})^{(1-y_{i})}$
##### Multi-class classification
- Softmax: $h(x_{i}) = \frac{e^{x_{i}}}{\sum_{j} e^{x_{j}}}$
- Outputs probability distribution
- $\sum_{i=1}^{K} h(x_{i}) = 1$ for $K$ classes
- Avoid exponentiating too large/small numbers for better stability: subtract $\mu$ max of $x_{i}$ $\Rightarrow$  $h(x_{i}) = \frac{e^{x_{i}-\mu}}{\sum_{j} e^{x_{j}-\mu}}$
- Loss: $-\sum_{j=1}^{K}y_{j}\log(p_{j})$

#### Universal Approximation Theorem
Feedforward networks with hidden layers provide a universal approximation framework.
A large MLP with even a single hidden layer is able to represent any function provided
that the network is given enough hidden units.
- No guarantee that the training algorithm will learn the function.
- In worse case, shallow one hidden layer network requires an exponential number of hidden units to have same approximation power as a deep net.

#### Backpropagation
For practice review assignment 1.
![[autodiff.png]]

Jacobian: generalization of the gradient for vector-valued functions $h(x)$. All input dimensions contribute to all output dimensions.
![[jacobian.png | center | 300]]
The Jacobian represents a local linearization of a function given a coordinate.
- Not unlike derivative being the best linear approximation of a curve (tangent).
The Jacobian determinant (for square matrices) measures the ratio of areas.
- Similar to what the ‘absolute slope’ measures in the 1d case (derivative).




