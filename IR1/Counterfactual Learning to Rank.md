## Introduction to Counterfactual Learning to Rank

Counterfactual Learning to Rank (CLTR) is an advanced approach to training ranking systems using historical user interaction data while accounting for biases inherent in how users interact with rankings. This lecture by Harrie Oosterhuis (Radboud University, 2025) covers the theoretical foundations and practical applications of this important field.

### Key Goals of the Lecture:

1. Understand biases in user interactions with rankings
2. Learn the fundamentals of counterfactual learning to rank
3. Explore the mathematical foundation
4. Overview advanced approaches
5. Examine real-world results and impacts

## Background: Learning to Rank & Online Evaluation

### Learning to Rank Pipeline

The standard ranking pipeline consists of several stages:

6. Documents (content items) that need to be ranked
7. Feature extraction or embedding lookup for these items
8. Scoring function that assigns relevance scores
9. Sorting based on these scores
10. Final presentation to users

### Traditional Learning to Rank Methods

Traditional supervised Learning to Rank (LTR) methods rely on labeled datasets with expert-created relevance judgments. They typically use:

- Pointwise approaches (predict relevance for individual items)
- Pairwise approaches (predict relative order between pairs of items)
- Listwise approaches (optimize entire ranking directly)

### Limitations of Supervised Approach

While supervised LTR has advantages (reliable ground truth, control, consistency), it suffers from significant limitations:

- Expensive and time-consuming to create annotated datasets
- Stationary (doesn't adapt to changing user preferences)
- Unethical in privacy-sensitive settings
- Impossible for personalization with unknown relevances
- Often disagrees with actual user preferences

### Online Evaluation Methods

Online controlled experiments like A/B testing and interleaving are used to evaluate ranking systems:

- A/B testing: compare systems with different user groups
- Interleaving: combine rankings from two systems for direct comparison

However, these methods have downsides:

- Users may be exposed to inferior systems
- Time-consuming (at least a week per comparison)

This raises the question: **How can we optimize and evaluate ranking systems on historical user interactions without running new online experiments?**

## Position Bias in User Interactions

### Position Bias Phenomenon

The display position of an item strongly affects user interactions due to:

- Users not examining all ranks (attention diminishes with position)
- Higher trust in items at the top positions
- Sometimes only top-ranked items being displayed

This means click-through-rates (CTRs) often tell us more about how an item was displayed than how relevant it actually is.

### Position Bias as Selection Bias

Position bias can be modeled as a selection bias in the rank-based examination click model:

- Examination as a selection filter per rank
- Mathematical model: $P(C=1|k,d,q) = P(E=1|k,q) \cdot P(R=1|d,q)$

Where:

- $P(C=1|k,d,q)$ is probability of clicking item $d$ at position $k$ for query $q$
- $P(E=1|k,q)$ is probability of examining position $k$ for query $q$
- $P(R=1|d,q)$ is probability of item $d$ being relevant to query $q$

This formula shows that clicks depend both on examination (influenced by position) and relevance.

## Naive Estimation Methods

When evaluating a new ranking policy $y$ using historical data, we want to know its Discounted Cumulative Gain (DCG):

$$U(y) = \sum_{k=1}^{K} \frac{P(R=1|y[k],q)}{\log_2(k+1)}$$

Where $y[k]$ is the item at position $k$ in ranking $y$.

### Naive Approach

A naive approach assumes click frequency equals relevance:

$$\hat{R}_{\text{Naive}}(d) = \frac{1}{N}\sum_{i=1}^{N}c_i(d)$$

$$\hat{U}_{\text{Naive}}(y) = \sum_{k=1}^{K}\frac{\hat{R}_{\text{Naive}}(y[k])}{\log_2(k+1)}$$

This leads to a biased estimate because:

$$\mathbb{E}_c[\hat{R}_{\text{Naive}}(d)] = \mathbb{E}_{y\sim\pi_0}[P(E=1|\text{rank}(d|y),q)]P(R=1|d,q)$$

Where $\pi_0$ is the logging policy that generated the historical data.

The problem is that the expected probability of examination under the logging policy affects our relevance estimates.

## Inverse Propensity Scoring (IPS)

### IPS Concept and Application

Inverse Propensity Scoring addresses selection bias by:

- Weighting observations inversely to their observation probability
- Correcting for position bias in rankings

The IPS estimator for relevance is:

$$\hat{R}_{\text{IPS}}(d) = \frac{1}{N}\sum_{i=1}^{N}\frac{c_i(d)}{P(E=1|\text{rank}(d|y_i),q)}$$

And for evaluating a new ranking:

$$\hat{U}_{\text{IPS}}(y) = \sum_{k=1}^{K}\frac{\hat{R}_{\text{IPS}}(y[k])}{\log_2(k+1)}$$

This estimate is unbiased because:

$$\mathbb{E}_c[\hat{R}_{\text{IPS}}(d)] = \mathbb{E}_{y\sim\pi_0}\left[\frac{P(E=1|\text{rank}(d|y),q)P(R=1|d,q)}{P(E=1|\text{rank}(d|y),q)}\right] = P(R=1|d,q)$$

### IPS Challenges

While unbiased, IPS can have high variance:

- Low propensities lead to large inverse weights
- Noise in a few observations gets amplified

One solution is propensity clipping:

$$\hat{R}_{\text{Clipped-IPS}}(d) = \frac{1}{N}\sum_{i=1}^{N}\frac{c_i(d)}{\max(P(E=1|\text{rank}(d|y_i),q),\tau)}$$

where $\tau$ is a clipping threshold that introduces some bias to reduce variance.

## Position Bias Estimation

To apply IPS, we need accurate estimates of position bias (examination probabilities). Several methods exist:

### Randomization Methods

11. **RandTop-k**: Shuffle the top-k results
    
    - The mean CTR per rank after randomization is proportional to position bias
    - However, this can negatively impact user experience
12. **RandPair**: Swap a random document with one at a fixed rank k
    
    - Less disruptive to user experience
    - The CTR ratio between original and swapped positions is proportional to the position bias ratio
13. **FairPairs**: Swap neighboring pairs
    
    - Even more subtle intervention

### Intervention Harvesting

Instead of deliberate randomization, we can leverage natural variation:

- Use data from previous A/B tests where different ranking systems showed items at different positions
- Collect "interventional sets" of items that different rankers placed in different positions
- Weight clicks by their exposure at different ranks

## Click Models for Relevance Estimation

### Jointly Estimating Bias and Relevance

Click models predict user behavior while simultaneously estimating both position bias and document relevance:

14. **Probabilistic Click Models**: Model user behavior with bias and relevance as parameters
15. **RegressionEM**: Extends click models to use query-document features for relevance estimation
16. **Neural Click Models / Two-Tower Models**: Neural networks that separate relevance and bias estimation

Two-tower models explicitly separate:

- The relevance part (using only document features)
- The bias part (using only contextual features)

These models optimize for the likelihood of observed interaction data rather than using IPS.

### Advantages and Limitations of Click Models

Click models don't provide strong guarantees about unbiasedness, but they:

- Are relatively easy to apply
- Have lower variance than IPS
- Require fewer assumptions

Whether maximizing likelihood leads to correct estimates depends on:

- Randomization in the logged rankings
- Availability of informative features
- Having enough observations
- Whether the optimization finds the right minimum

## Advanced Estimation Methods

### The Direct Method (DM)

The Direct Method uses a model trained on features to directly estimate relevance:

- No selection problem (estimates relevance for every document)
- Low variance (no reweighting of interactions)
- But biased unless the regression model is perfectly correct

### Doubly Robust Estimation (DR)

Doubly Robust estimation combines the best of both approaches:

- Starts with the Direct Method estimate
- Uses IPS to correct for biases in the DM's predictions
- Unbiased if either IPS is unbiased OR DM is unbiased
- Generally has lower variance than IPS alone

The DR approach subtracts an unbiased IPS estimate of the DM's bias from the DM estimate.

## Real-World Results and Conclusion

### Online A/B Test Results

Most research in unbiased learning to rank relies on simulation experiments that show considerable performance gains. However, real-world A/B tests typically show more modest improvements:

- The median uplift in online A/B tests is around 2.08%
- While significant, this is less dramatic than simulation results might suggest

### Ongoing and Future Research Directions

17. **Fairness in ranking**: Distributing exposure in a meritocratic manner
18. **Safety in LTR**: Ensuring new models don't decrease production performance
19. **Uncertainty and Bayesian approaches**: Moving beyond frequentist methods
20. **Other forms of bias**: Trust, item-selection, surrounding-items effects
21. **Other interaction types**: Conversion, purchases, watch-time, likes
22. **Long-term objectives**: Going beyond immediate outcomes

## Summary of Key Points

23. User interactions with rankings are valuable but biased, especially by position
24. Position bias can be estimated through randomization or intervention harvesting
25. Inverse Propensity Scoring (IPS) can unbiasedly estimate relevance from clicks
26. Click models jointly estimate bias and relevance through likelihood maximization
27. Doubly Robust estimation combines direct regression with IPS correction
28. Counterfactual LTR makes real-world differences, though often modest in magnitude

The field continues to evolve with applications in search engines, recommendation systems, and other ranking contexts where learning from user feedback is essential.