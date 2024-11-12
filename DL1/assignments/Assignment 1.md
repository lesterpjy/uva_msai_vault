#### 1. a)
$$
\begin{equation}
\begin{aligned}
 \frac{\partial L}{\partial \mathbf{W}} &= \left( \frac{\partial L}{\partial \mathbf{Y}} \right)^{\top} \frac{\partial \mathbf{Y}}{\partial \mathbf{W}}
\end{aligned}
\end{equation}
$$
where $\frac{\partial L}{\partial W} \in \mathbb{R}^{N \times M}$ and $\frac{\partial L}{\partial Y} \in \mathbb{R}^{S \times N}$, so $\frac{\partial Y}{\partial W} \in \mathbb{R}^{S \times M}$   
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial W_{ij}} &= \sum_{a}^{S} \sum_{b}^{N} \frac{\partial L}{\partial Y_{ab}} \frac{\partial Y_{ab}}{\partial W_{ij}}
\end{aligned}
\end{equation}
$$
and
$$
\begin{equation}
\begin{aligned}
\frac{\partial Y_{ab}}{\partial W_{ij}} 
&= \frac{\partial}{\partial W_{ij}} \left( \sum_{k} X_{ak} W_{kb}^{\top} + B_{ab} \right) \\
&= \sum_{k} X_{ak} \frac{\partial W_{bk}}{\partial W_{ij}} \\
&= \sum_{k} X_{ak} \delta_{ib} \delta_{jk} \\
&= X_{aj}\delta_{ib}
\end{aligned}
\end{equation}
$$
Substituting in the partial derivative,
$$
\begin{equation}
\begin{aligned}
\sum_{a}^{S} \sum_{b}^{N} \frac{\partial L}{\partial Y_{ab}} \left( X_{aj} \delta_{ib}\right) &= \sum_{a}^S \frac{\partial L}{\partial Y_{ai}} X_{aj} \\ 
\Rightarrow 
\frac{\partial L}{\partial W_{ij}} &= \left( \left[ \frac{\partial L}{\partial Y}\right]_{:,i}\right)^{\top} X_{:,j} \\
\Rightarrow
\frac{\partial L}{\partial W} &= \left(\frac{\partial L}{\partial Y}\right)^{\top} X
\end{aligned}
\end{equation}
$$

#### 1. b)
$$
\frac{\partial L}{ \partial b_{i}} = \sum_{a}^S \sum_{b}^{N} \frac{\partial L}{\partial Y_{ab}} \frac{\partial Y_{ab}}{\partial b_{i}}
$$
and
$$
\frac{\partial Y_{ab}}{\partial b_{i}} = \frac{\partial}{\partial b_{i}} \left( \sum_{k} X_{{ak}} W_{kb}^{\top} + B_{ab} \right) = \delta_{bi}
$$
Substituting in the partial derivative,
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{ \partial b_{i}} = \sum_{a}^{S} \sum_{b}^{N} \frac{\partial L}{\partial Y_{ab}} \delta_{bi} &= \sum_{a}^S \frac{\partial L}{\partial Y_{ai}} \\
\Rightarrow \frac{\partial}{\partial \mathbf{b}} = \sum_{a}^S \frac{\partial L}{\partial Y_{a,:}} = \mathbb{1} \frac{\partial L}{\partial Y}
\end{aligned}
\end{equation}
$$
where $\mathbb{1} \in \mathbb{R}^{1\times S}$ and $\frac{\partial L}{\partial Y} \in \mathbb{R}^{S \times N}$.

#### 1. c)
$$
\begin{equation}
\begin{aligned}
 \frac{\partial L}{\partial X} &= \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X}
\end{aligned}
\end{equation}
$$
where $\frac{\partial L}{\partial X} \in \mathbb{R}^{S \times M}$ and $\frac{\partial L}{\partial Y} \in \mathbb{R}^{S \times N}$, so $\frac{\partial Y}{\partial X} \in \mathbb{R}^{N \times M}$ 
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial X_{ij}} &= \sum_{a}^{S} \sum_{b}^{N} \frac{\partial L}{\partial Y_{ab}} \frac{\partial Y_{ab}}{\partial X_{ij}}
\end{aligned}
\end{equation}
$$
and
$$
\begin{equation}
\begin{aligned}
\frac{\partial Y_{ab}}{\partial X_{ij}} &= \frac{\partial}{\partial X_{ij}} \left( \sum_{k} X_{ak} W_{kb}^{\top} + B_{ab} \right) \\
&= \sum_{k} \frac{\partial X_{ak}}{\partial X_{ij}} W_{bk} = \sum_{k} \delta_{ia} \delta_{jk} W_{bk} = W_{bj} \delta_{ia}
\end{aligned}
\end{equation} 
$$
Substituting in the partial derivative,
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial X_{ij}} &= \sum_{a}^{S} \sum_{b}^{N} \frac{\partial L}{\partial Y_{ab}} W_{bj} \delta_{ia} = \sum_{b}^N \frac{\partial L}{\partial Y_{ib}} W_{bj} \\
\Rightarrow
\frac{\partial L}{\partial X_{ij}} &= \left( \left[ \frac{\partial L}{\partial Y} \right]_{i,:}\right) W_{:,j} \\
\Rightarrow
\frac{\partial L}{\partial X} &= \frac{\partial L}{\partial Y} W
\end{aligned}
\end{equation}
$$

#### 1. d)
Given $Y = h(X)$ and $Y_{ij} = h(X_{ij})$, 
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{X} \Rightarrow \frac{\partial L}{\partial X_{ij}} = \sum_{a}^S \sum_{b}^N \frac{\partial L}{\partial Y_{ab}} \frac{\partial Y_{ab}}{\partial X_{ij}}
$$
First evaluating $\frac{\partial Y_{ab}}{\partial X_{ij}}$:
$$
\begin{equation}
\begin{aligned}
\frac{\partial Y_{ab}}{\partial X_{ij}} &= \frac{\partial}{\partial X_{ij}}\left( h(X_{ab}) \right) = h'(X_{ab}) \frac{\partial X_{ab}}{\partial X_{ij}} = h'(X_{ab})\delta_{ia}\delta_{jb}
\end{aligned}
\end{equation}
$$
Substituting in the partial derivative,
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial X_{ij}} &= \sum_{a}^S \sum_{b}^N \frac{\partial L}{\partial Y_{ab}}h'(X_{ab})\delta_{ia}\delta_{jb} = \frac{\partial L}{\partial Y_{ij}}h'(X_{ij}) \\
\Rightarrow \frac{\partial L}{\partial X} &= \frac{\partial L}{\partial Y} \circ h'(X)
\end{aligned}
\end{equation}
$$
#### 1. e)
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial Z} &= Y \circ \left( \frac{\partial L}{\partial Y} - \left( \frac{\partial L}{\partial Y} \circ Y\right) \mathbb{1}\mathbb{1}^{\top}\right) \\
&= Y \circ \left( \left( - \frac{1}{S} \frac{T}{Y}\right) - \left( - \frac{1}{S} \frac{T}{Y} \circ Y\right) \mathbb{1}\mathbb{1}^{\top}\right) \\
&= \left( - \frac{1}{S} T\right)  + \left(\frac{1}{S} T\circ Y\right) \mathbb{1}\mathbb{1}^{\top} \\
&= \frac{1}{S} \left( Y \circ T \ \mathbb{1}\mathbb{1}^{\top} -T\right)\\
&= \frac{1}{S} (Y \circ \mathbb{1} - T) \\
&= \frac{1}{S} (Y - T)
\end{aligned}
\end{equation}
$$
Therefore, $\alpha = \frac{1}{S}$ and $M = (Y-T)$

#### 4. a) Show that the eigenvalues for the Hessian matrix in a strictly local minimum are all positive.

For a continuously differentiable function $f$ and real non-singular (invertible) Hessian matrix $H$ at point $x_{p}$ , we can perform second order Taylor expansion at $x_{p}$ for sufficiently small $h$:
$$
f(x_{p}​+h)=f(x_{p}​)+ \nabla f(x_{p})^\top h + \frac{1}{2}h^\top H(x_{p})h + o(||h||^2)
$$
At the critical point $x_{p}$, we have $\nabla f(x_{p}) = 0$, and thus the Taylor expansion gives,
$$
f(x_{p}​+h) - f(x_{p}​) = \frac{1}{2}h^\top H(x_{p})h
$$
for sufficiently small $h$ where the higher order terms $o(||h||^2)$ is negligible.

In a strictly local minimum with critical point $x_{p}$ the function should satisfy:
$$
f(x_{p} + h) - f(x_{p}) > 0
$$
for $h \neq 0$. Substituting the term obtained with Taylor expansion above, we have,
$$
\frac{1}{2}h^\top H(x_{p})h > 0 \ \ \ \text{for all} \ h \neq 0 
$$
This inequality implies that the symmetric real Hessian matrix $H$ is positive definite for all non-zero $h$. The matrix $H$ is also positive definite if and only if all of its eigenvalues are positive. We now prove that positive definiteness implies that the eigenvalues are strictly positive.
Since $H$ is symmetric, we can diagonalize it as
$$
H = Q\Lambda Q^\top
$$
where $Q$ is the orthogonal matrix with the eigenvectors of $H$, and $\Lambda$ is a diagonal matrix containing the eigenvalues $\lambda_{i}$.
The quadratic form can then be re-expressed in its index notation:
$$
h^\top H h = h^\top Q\Lambda Q^\top h = (Q^\top h)^\top \Lambda (Q^\top h)
$$
$$
\Rightarrow h^\top H h  = \\\sum_{i=1}^n \lambda_{i}y_{i}^2 \ \ \text{where} \ \ y = (Q^\top h)
$$
Since $h^\top H h > 0$, for all $h \neq 0$, all eigenvalues $\lambda_{i}$ must be strictly positive.

#### 4. b) If some of the eigenvalues of the Hessian matrix at point p are positive and some are negative, this point would be a saddle point; intuitively explain why the number of saddle points is exponentially larger than the number of local minima for higher dimensions?

Each eigenvalues of the Hessian corresponds to the curvature of the function $f$ in the direction of the associated eigenvector. Based on 4(a) we proved that in a local minimum, all the eigenvalues of the Hessian are all positive and the function curves upward in a bowl shape (convex). Conversely for a local maximum, all the eigenvalues of the Hessian are negative and the function curves downward around the critical point in a concave shape.
In $n$-dimensions, the Hessian will have $n$ eigenvalues, each of which can be positive, negative, or zero. For simplicity using the coin flip analogy, the eigenvalue can be positive or negative (heads or tails), and thus the number possible configurations of the eigenvalues in $n$-dimensions is $2^n$.
There is only one configuration for the local maxima and local minima where the eigenvalues are strictly negative or strictly positive, respectively. The rest of the configurations are all mix of positive and negative values that give saddle points. The number of saddle point configurations is thus given by $2^n - 2$, in other terms, $O(2^n)$, exponentially larger than the number of local minima. 
#### 4. c) By using the update formula of gradient descent around saddle point p, show why saddle points can be harmful to training.

The gradient descent algorithm for a continuously differentiable function $f$ at point $x$ can be given by,
$$
x^{(k+1)} = x^{(k)} - \eta \nabla f(x^{(k)})
$$
where $x^{(k)}$ is the $n$-dimensional parameter vector at step $k$, and $\eta$ is the learning rate. At the saddle point $p$ where the Hessian $H(p)$ has both positive and negative values. Using Taylor expansion of the gradient around the saddle point $p$ for point $x$ we have,
$$
\nabla f(x) = \nabla f(p) + H(p)(x-p) + \dots
$$
Since at the saddle point $\nabla f(p) = 0$, the linear approximation becomes,
$$
\nabla f(x) \approx H(p)(x-p)
$$
The gradient update rule can thus be simplified as,
$$
x^{(k+1)} = x^{(k)} - \eta H(p)(x^{(k)} - p) 
$$
For some small difference around the saddle point $\delta^{(k)} = x^{(k)} - p$, then,
$$
\begin{equation}
\begin{aligned}
\delta^{(k+1)} &= x^{(k+1)} - p \\
&=x^{(k)} - \eta H(p)(x^{(k)}-p)-p \\
&=\delta^{(k)} - \eta H(p)(\delta^{(k)}) \\
\Rightarrow \delta^{(k+1)} &= (I - \eta H(p))(\delta^{(k)})
\end{aligned}
\end{equation}
$$
With diagonalization of the Hessian as in 4(a), $H(p) = Q\Lambda Q^\top$, the displacement $\delta^{(k)}$ can be transformed into the new coordinate of the eigenvectors of $H(p)$ with,
$$
\begin{equation}
\begin{aligned}
y^{(k)} &= Q^{\top} \delta^{(k)} \\
\Rightarrow \delta^{(k)} &= Q y^{(k)}
\end{aligned}
\end{equation}
$$
Substituting into the update equation,
$$
\begin{equation}
\begin{aligned}
\delta^{(k+1)} &= (I - \eta H(p))(\delta^{(k)}) \\
\Rightarrow \delta^{(k+1)} &= (I - \eta Q\Lambda Q^{\top})Q y^{(k)} \\
\Rightarrow \delta^{(k+1)} &= (IQ - \eta Q\Lambda Q^{\top}Q) y^{(k)} \\
\Rightarrow \delta^{(k+1)} &= (IQ - \eta Q\Lambda I) y^{(k)} \\
\Rightarrow \delta^{(k+1)} &= Q(I - \eta \Lambda) y^{(k)} \\
\end{aligned}
\end{equation}
$$
since $Q^{\top}Q = I$. To find $y^{(k+1)}$, we can $\delta^{(k+1)}$ project back into the eigenvector basis,
$$
y^{(k+1)} = Q^{\top}Q(I - \eta \Lambda) y^{(k)} = (I - \eta \Lambda) y^{(k)}
$$
Since the diagonal matrix $\Lambda$ contains the eigenvectors $\lambda_{i}$, the update is dependent on the sign of the eigenvalues. For each component $i$, the update rule is then,
$$
y_{i}^{(k+1)} = (1 - \eta \lambda_{i}) y_{i}^{(k)}
$$
This suggests that the behavior near a saddle point where the eigenvalues can be both positive and negative is harmful since each component of the parameter vector evolve independently. For positive eigenvalues, $y_{i}^{(k)}$ converges in a gradient descent given appropriate $\eta$. For negative eigenvalues,  $|(1 - \eta \lambda_{i})| > 1$, thus $y_{i}^{(k)}$ increases and leads to divergence. This means that the saddle point is harmful to gradient descent convergence and leads to oscillations or divergence.


#### 5. a) Adding batch normalization layers causes changes to the back propagation steps, because we also want to optimize the learnable $\beta_{i}$ and $\gamma_{i}$ parameters. Assume that we have already backpropagated up to the output of the batch norm node, and therefore we have each $\frac{\partial L}{\partial y_{i}}$, where $y_{i} = \gamma_{i}x_{i} + \beta_{i}$. Write the derivatives of $L$ loss with respect to the two parameters $\beta_{i}$ and $\gamma_{i}$ in terms of $\frac{\partial L}{\partial y_{i}}$.
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial\beta_{i}} &= \frac{\partial L}{\partial y_{i}} \frac{\partial y_{i}}{\partial \beta_{i}}\\
\Rightarrow \frac{\partial L}{\partial\beta_{i}} &= \frac{\partial L}{\partial y_{i}} \left\{\frac{\partial}{\partial \beta_{i}}  \gamma_{i}x_{i} + \beta_{i}\right\} \\
\Rightarrow \frac{\partial L}{\partial\beta_{i}} &= \frac{\partial L}{\partial y_{i}} \left\{1 \right\} \\
\Rightarrow \frac{\partial L}{\partial\beta_{i}} &= \frac{\partial L}{\partial y_{i}} \\
\end{aligned}
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial\gamma_{i}} &= \frac{\partial L}{\partial y_{i}} \frac{\partial y_{i}}{\partial \gamma_{i}}\\
\Rightarrow \frac{\partial L}{\partial\gamma_{i}} &= \frac{\partial L}{\partial y_{i}} \left\{\frac{\partial}{\partial \gamma_{i}}  \gamma_{i}x_{i} + \beta_{i}\right\} \\
\Rightarrow \frac{\partial L}{\partial\gamma_{i}} &= \frac{\partial L}{\partial y_{i}} \left\{ x_{i} \right\}
\end{aligned}
\end{equation}
$$
#### 5. b) Consider applying batch normalization to a fully connected layer with an input size of 20 and an output size of 40. How many training parameters does this layer have, including batch normalization parameters?

The fully connected layer has $[20 \times 40]$ number of weights and $40$ bias parameters for each of the output. Batch normalization introduces the additional $\beta_{i}$ and $\gamma_{i}$ parameters for each of the output. The total number of parameters that this layer has can thus be calculated as,
$$
([20\times 40] + 40) + 40 + 40 = 920
$$
#### 5. c) During training, batch normalization normalizes inputs using the mean and variance of the current mini-batch. Explain why it would be problematic to normalize inputs the same way during inference (test time), and how batch normalization addresses this problem.

Usually at test time, the model can often be fed with test data of differing sizes, or even just one sample at a time, rather than the mini-batch size used during training. Therefore, if the model was trained with a mini-batch size of 32, meaning the training statistics of mean and variance will depend on this size, variable batch-sizes will not be able to be used for testing to maintain consistent estimation of the statistics. Additionally, at test time we seek deterministic output from the model, and using the current mini-batch statistics for normalization during testing would mean the output is dependent of the batch, leading to inconsistent predictions.
Batch normalization addresses this problem by maintaining moving averages of the mean and variance for each batch-normalize layer during training. This helps the model estimate the statistics over the entire dataset. Then at test time, the batch-normalize layers then use these statistics for normalization.

#### 5. d) Experimental analysis showed that a high percentage of neurons are dead in networks with ReLU activation functions (you can refer to tutorial 3 for more information). Explain the concept of a dead neuron, when it occurs when using ReLU, and how it harms training.

A dead neuron is a neuron in a neural network that consistently outputs zero, effectively becoming inactive or "dead" during training. This implies that no gradient is provided and the parameter of the neuron cannot be trained in the previous layer to obtain non-zero output. As ReLU is defined as $ReLU(x) = max(0, x)$, if the input $x$ to the function is less than or equal to zero, the output of ReLU is zero. Dead neurons can occur in this case when the updates to the network parameters causes the input to consistently be in the range of $x \leq 0$, from which the neuron might never recover and the neuron will be considered "dead".
This is harmful to network training because the dead neuron does not contribute to the modeling or learning capability of the network, and if a significant number of neurons becomes inactive, the model may underfit or fail to generalize at test time.

#### 5. d) How does batch normalization prevent neurons from dying?

Batch normalization prevents neuron from dying by controlling the activation range of the input to each layer and reducing the internal covariate shift. By keeping the activations close to the mean of the mini-batch, and minimizing the shift of the distributions of layer inputs as the parameters of the previous layers change, batch normalization prevents ReLU activations from consistently receiving negative values as inputs, thereby preventing neurons from dying.


