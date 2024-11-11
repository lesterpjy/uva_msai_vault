
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



