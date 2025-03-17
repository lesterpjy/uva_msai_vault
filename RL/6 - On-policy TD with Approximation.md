
This lecture covers how to extend temporal difference (TD) learning to use function approximation, which is crucial for handling large or continuous state spaces in reinforcement learning. I'll walk through the key concepts covered in the slides.

## Function Approximation in Reinforcement Learning

In previous lectures, you learned about tabular methods where we store a separate value for each state or state-action pair. However, this becomes impractical when dealing with large state spaces. Function approximation allows us to generalize across states by using parametrized functions.

### Linear Function Approximation

A linear function approximator represents the value function as:

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T\mathbf{x}(s) = \sum_{i=1}^{d} w_i x_i(s)$$

Where:

- $\mathbf{w}$ is a parameter vector with elements $w_i$
- $\mathbf{x}(s)$ is a feature vector for state $s$ with elements $x_i(s)$
- $d$ is the number of features

It's important to note that while the value function is linear in the parameters $\mathbf{w}$, it is not necessarily linear in the state $s$. The features $\mathbf{x}(s)$ can perform non-linear transformations of the state.

Common feature types include:

- Polynomial features (linear, quadratic, etc.)
- State aggregation (one-hot encoding of state partitions)
- Radial basis functions (Gaussian-like features centered at points in state space)
- Task-specific features based on domain knowledge

A key insight: the tabular case is actually a special case of linear function approximation where features are one-hot encodings of individual states.

### Non-linear Function Approximation

Non-linear function approximators cannot be expressed in the form $\mathbf{w}^T\mathbf{x}(s)$. The most common example is neural networks, where parameters interact non-linearly through activation functions. For example, a simple feedforward neural network might compute:

$$\hat{v}(s, \mathbf{W}^{(1)}, \mathbf{W}^{(2)}) = \sum_{m=0}^{M} w_m^{(2)} h^{(1)}\left(\sum_{d=0}^{D} w_{md}^{(1)}s_d\right)$$

Where $h^{(1)}$ is a non-linear activation function.

Modern deep RL can use various neural network architectures:

- Feedforward networks
- Convolutional neural networks
- Graph neural networks
- Transformers

## Gradient Monte Carlo with Function Approximation

Recall that in gradient Monte Carlo, we update parameters to minimize the value error by using gradient descent:

$$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \alpha [G_t - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)$$

Where:

- $G_t$ is the actual return
- $\alpha$ is the step size
- $\nabla\hat{v}(S_t, \mathbf{w}_t)$ is the gradient of the value function with respect to parameters

For linear function approximation, this gradient is simply:

$$\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}) = \nabla_{\mathbf{w}}(\mathbf{w}^T\mathbf{x}(S_t)) = \mathbf{x}(S_t)$$

Making the update rule:

$$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \alpha [G_t - \hat{v}(S_t, \mathbf{w}_t)]\mathbf{x}(S_t)$$

This is very computationally efficient. For non-linear approximators like neural networks, the gradient is more complex and typically computed using backpropagation.

## Semi-gradient TD Learning

The natural extension of TD learning to function approximation replaces the Monte Carlo return $G_t$ with the TD target $R_t + \gamma\hat{v}(S_{t+1}, \mathbf{w}_t)$:

$$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \alpha [R_t + \gamma\hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)$$

This is called a "semi-gradient" method because it ignores that the target $\hat{v}(S_{t+1}, \mathbf{w}_t)$ also depends on $\mathbf{w}$. The true gradient would include the term $\nabla_{\mathbf{w}}[\gamma\hat{v}(S_{t+1}, \mathbf{w}_t)]$, but using this full gradient tends to perform poorly in practice.

### The TD Fixed Point

For linear function approximation with semi-gradient TD, the parameters converge to what's called the "TD fixed point" $\mathbf{w}_{\text{TD}}$. To find this fixed point analytically:

1. Rewrite the TD update as an expected update: $$E[\mathbf{w}_{t+1}|\mathbf{w}_t] = \mathbf{w}_t + \alpha\left(E[R_{t+1}\mathbf{x}_t] - E[\mathbf{x}_t(\mathbf{x}_t - \gamma\mathbf{x}_{t+1})^T]\mathbf{w}_t\right)$$
    
2. At convergence, $E[\mathbf{w}_{t+1}|\mathbf{w}_t] = \mathbf{w}_t$, which gives: $$E[R_{t+1}\mathbf{x}_t] - E[\mathbf{x}_t(\mathbf{x}_t - \gamma\mathbf{x}_{t+1})^T]\mathbf{w}_{\text{TD}} = 0$$
    
3. If we define:
    
    - $\mathbf{b} = E[R_{t+1}\mathbf{x}_t]$
    - $\mathbf{A} = E[\mathbf{x}_t(\mathbf{x}_t - \gamma\mathbf{x}_{t+1})^T]$
4. Then $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1}\mathbf{b}$
    

Importantly, the TD fixed point is generally not the same as the minimum of the value error that gradient MC would find. However, semi-gradient TD tends to learn faster than gradient MC due to its lower variance, making it often preferred in practice despite not finding the global optimum.

### Convergence Guarantees

- Gradient Monte Carlo with linear function approximation converges to a global minimum of the value error
- Semi-gradient TD with linear function approximation converges to the TD fixed point
- With non-linear function approximation, gradient MC converges to a local minimum of value error
- With non-linear function approximation, semi-gradient TD may diverge (no convergence guarantee)

## Least-Squares Temporal-Difference (LSTD)

LSTD is a method that directly computes the TD fixed point without iterative updates. Since $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1}\mathbf{b}$, LSTD estimates $\mathbf{A}$ and $\mathbf{b}$ from data and then computes $\mathbf{w}_{\text{TD}}$ directly:

$$\hat{\mathbf{A}}_t = \sum_{k=0}^{t-1} \mathbf{x}_k(\mathbf{x}_k - \gamma\mathbf{x}_{k+1})^T + \varepsilon\mathbf{I}$$ $$\hat{\mathbf{b}}_t = \sum_{k=0}^{t-1} R_{k+1}\mathbf{x}_k$$ $$\mathbf{w}_t = \hat{\mathbf{A}}_t^{-1}\hat{\mathbf{b}}_t$$

The $\varepsilon\mathbf{I}$ term is a regularization term to ensure $\hat{\mathbf{A}}_t$ is invertible.

LSTD has several properties:

- More sample efficient than semi-gradient TD
- More computationally intensive (naive matrix inversion is O(d³), but can be done incrementally in O(d²))
- No step size parameter needed (though ε is still a hyperparameter)
- Never "forgets" data, which is good for stationary problems but problematic if the environment changes

## Control with Function Approximation

The principles of function approximation can also be applied to action-value functions for control. For semi-gradient SARSA, the update rule becomes:

$$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \alpha [R_t + \gamma\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{q}(S_t, A_t, \mathbf{w}_t)$$

Using an ε-greedy policy derived from the current approximation of q, we can learn control policies for large state spaces.

## Summary of Covered Material

- Linear function approximation: uses feature vectors to represent states
- Non-linear function approximation: primarily neural networks
- Semi-gradient TD: extends TD learning to function approximation
- The TD fixed point: what semi-gradient TD converges to
- Least-squares TD: directly computes the TD fixed point
- Semi-gradient SARSA: extends function approximation to control

The next lectures will cover off-policy TD learning with function approximation, which introduces additional challenges not present in the on-policy case.

Function approximation is a critical component of modern reinforcement learning that allows these methods to scale to real-world problems with large or continuous state spaces.