## Introduction to Counterfactual Learning to Rank

Counterfactual Learning to Rank (CLTR) is an advanced approach to training ranking systems using historical user interaction data while accounting for biases inherent in how users interact with rankings.

## Background: Learning to Rank & Online Evaluation

### Learning to Rank Pipeline

The standard ranking pipeline consists of several stages:

1. Documents (content items) that need to be ranked
2. Feature extraction or embedding lookup for these items
3. Scoring function that assigns relevance scores
4. Sorting based on these scores
5. Final presentation to users

### Traditional Learning to Rank Methods

Traditional supervised Learning to Rank (LTR) methods <span style="color:rgb(172, 96, 230)">rely on labeled datasets with expert-created relevance judgments</span>. They typically use:

- Pointwise approaches (predict relevance for individual items)
- Pairwise approaches (predict relative order between pairs of items)
- Listwise approaches (optimize entire ranking directly)

### Limitations of Supervised Approach

While supervised LTR has <span style="color:rgb(172, 96, 230)">advantages (reliable ground truth, control, consistency)</span>, it suffers from significant limitations:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Expensive and time-consuming to create annotated datasets
- Stationary (doesn't adapt to changing user preferences)
- Unethical in privacy-sensitive settings
- Impossible for personalization with unknown relevances
- Often disagrees with actual user preferences

### Online Evaluation Methods

Online controlled experiments like A/B testing and interleaving are used to evaluate ranking systems:

- <span style="color:rgb(172, 96, 230)">A/B testing</span>: compare systems with different user groups
- <span style="color:rgb(172, 96, 230)">Interleaving</span>: combine rankings from two systems for direct comparison

However, these methods have downsides:

- Users may be <span style="color:rgb(172, 96, 230)">exposed to inferior systems</span>
- <span style="color:rgb(172, 96, 230)">Time-consuming</span> (at least a week per comparison)

This raises the question: **How can we optimize and evaluate ranking systems on historical user interactions without running new online experiments?**

## Position Bias in User Interactions

### Position Bias Phenomenon

The display position of an item strongly affects user interactions due to:

- Users not examining all ranks (attention diminishes with position)
- Higher trust in items at the top positions
- Sometimes only top-ranked items being displayed

<span style="color:rgb(172, 96, 230)">Position bias means click-through-rates (CTRs) tell us more about how an item was displayed than how relevant it actually is.</span>

### Position Bias as Selection Bias

Position bias can be modeled as a selection bias in the rank-based examination click model:

- <span style="color:rgb(172, 96, 230)">Examination as a selection filter per rank</span>
- Position-based model (PBM): $P(C=1|k,d,q) = P(E=1|k,q) \cdot P(R=1|d,q)$

Where:

- $P(C=1|k,d,q)$ is probability of clicking item $d$ at position $k$ for query $q$
- $P(E=1|k,q)$ is probability of examining position $k$ for query $q$
- $P(R=1|d,q)$ is probability of item $d$ being relevant to query $q$

This formula shows that clicks depend both on examination (influenced by position) and relevance.

## Naive Estimation Methods

When evaluating a new ranking policy $y$ using historical data, we want to know its Discounted Cumulative Gain (DCG):

$$U(y) = \sum_{k=1}^{K} \frac{P(R=1|y[k],q)}{\log_2(k+1)}$$

Where $y[k]$ is the item at position $k$ in ranking $y$.
However the available data is gathered previously on a production system.
Counterfactual evaluation: evaluate a ranking on interactions with other rankings.

### Naive Approach

A naive approach assumes <span style="color:rgb(172, 96, 230)">click frequency equals relevance</span>:

$$\hat{R}_{\text{Naive}}(d) = \frac{1}{N}\sum_{i=1}^{N}c_i(d)$$

$$\hat{U}_{\text{Naive}}(y) = \sum_{k=1}^{K}\frac{\hat{R}_{\text{Naive}}(y[k])}{\log_2(k+1)}$$

This leads to a biased estimate:

$$\mathbb{E}_c[\hat{R}_{\text{Naive}}(d)] = \mathbb{E}_{y\sim\pi_0}[P(E=1|\text{rank}(d|y),q)]P(R=1|d,q)$$

Where $\pi_0$ is the logging policy that generated the historical data.

The problem is that the <span style="color:rgb(172, 96, 230)">expected probability of examination under the logging policy affects our relevance estimates: the estimate is affected by both the true relevance AND the examination probability under the logging policy</span>

Explanation:
- **The data collection process**: User interactions (clicks) are collected from a previous system that showed results in a particular order.
- **Position bias effect**: Items shown higher in the ranking receive more attention and clicks regardless of their actual relevance.
- **Selection bias**: The previous system (logging policy) determined which documents were shown to users, creating a biased sample of interactions.
- **Counterfactual estimation**: The process of using this biased historical data to estimate how a new ranking would perform if deployed.

<span style="color:rgb(172, 96, 230)">Issues</span>:
- **Biased evaluation**: A new ranking system might appear worse than it actually is because items it would rank highly weren't shown often in the historical data.
- **Optimization problems**: If we train a new model using this biased data, it will learn to reproduce the biases of the old system rather than optimizing for true relevance.
- **Matthew effect**: Popular items get more exposure, leading to more clicks, which makes them appear more relevant, creating a feedback loop.
- **Cold start issues**: New or rare items with little historical interaction data will be undervalued.
## Inverse Propensity Scoring (IPS)

### IPS Concept and Application

<span style="color:rgb(172, 96, 230)">Inverse Propensity Scoring</span> addresses selection bias by:

- <span style="color:rgb(172, 96, 230)">Weighting observations inversely to their observation probability</span>
- In expectation, estimate not affected by any selection
- underlying idea same as importance sampling

The IPS estimator for relevance is:

$$\hat{R}_{\text{IPS}}(d) = \frac{1}{N}\sum_{i=1}^{N}\frac{c_i(d)}{P(E=1|\text{rank}(d|y_i),q)}$$

And for evaluating a new ranking:

$$\hat{U}_{\text{IPS}}(y) = \sum_{k=1}^{K}\frac{\hat{R}_{\text{IPS}}(y[k])}{\log_2(k+1)}$$

This estimate is unbiased because:

$$\mathbb{E}_c[\hat{R}_{\text{IPS}}(d)] = \mathbb{E}_{y\sim\pi_0}\left[\frac{P(E=1|\text{rank}(d|y),q)P(R=1|d,q)}{P(E=1|\text{rank}(d|y),q)}\right] = P(R=1|d,q)$$

### IPS Challenges

While unbiased, IPS can have high variance:

- <span style="color:rgb(172, 96, 230)">Low propensities lead to large inverse weights</span>
- <span style="color:rgb(172, 96, 230)">Noise in a few observations gets amplified -> high varaince</span>

One solution is <span style="color:rgb(172, 96, 230)">propensity clipping</span>:

$$\hat{R}_{\text{Clipped-IPS}}(d) = \frac{1}{N}\sum_{i=1}^{N}\frac{c_i(d)}{\max(P(E=1|\text{rank}(d|y_i),q),\tau)}$$

where $\tau$ is a <span style="color:rgb(172, 96, 230)">clipping threshold that introduces some bias to reduce variance</span>.

## Position Bias Estimation

To apply IPS, we need accurate estimates of position bias (examination probabilities). 

Assumptions of PBM:
- Examination hypothesis: Users click only on examined and relevant items
- Item relevance independent of other items
- observation depends only on the item's position
### Randomization Methods
From the assumption: only effect on clicks when showing an item at different rank is due to position bias.

Online intervention using randomization
1. **RandTop-k**: Shuffle the top-k results
    - The <span style="color:rgb(172, 96, 230)">mean CTR per rank after randomization is proportional to position bias</span>
    - However, this can <span style="color:rgb(172, 96, 230)">negatively impact user experience</span>
2. **RandPair**: Swap a random document with one at a fixed pivot rank k
    - Less disruptive to user experience
    - The <span style="color:rgb(172, 96, 230)">CTR ratio between original and swapped positions is proportional to the position bias ratio</span>

Limitation: still potential for relevant results to be swapped to lower positions
Can we avoid online interventions?
- <span style="color:rgb(172, 96, 230)">use variability in historic log data from multiple rankers to estimate position bias</span>

### Intervention Harvesting

Instead of deliberate randomization, we can <span style="color:rgb(172, 96, 230)">leverage natural variation</span>:

- Use data from previous A/B tests where different ranking systems showed items at different positions
- <span style="color:rgb(172, 96, 230)">Collect "interventional sets" of items that different rankers placed in different positions</span>
- <span style="color:rgb(172, 96, 230)">Weight clicks by their exposure at different ranks</span>
- Assumption: different rankers must not handle different queries or users

### Summary of estimation methods

Randomizing results online
+ <span style="color:rgb(146, 208, 80)">Gold standard for bias estimation</span>
- <span style="color:rgb(255, 82, 82)">Might harm user experience</span>

Intervention Harvesting
+ <span style="color:rgb(146, 208, 80)">Leverages variability in previously collected data</span>
- <span style="color:rgb(255, 82, 82)">Assumes no shift in the query distribution between rankers</span>
- <span style="color:rgb(255, 82, 82)">Assumes no shift of item relevance over time</span>

## Click Models for Relevance Estimation

### Jointly Estimating Bias and Relevance

Click models predict user behavior while simultaneously <span style="color:rgb(172, 96, 230)">estimating both position bias and document relevance</span>:

1. <span style="color:rgb(172, 96, 230)">Probabilistic Click Models</span>
	- Predict user click behavior with bias and relevance as parameters
	- PBM: early click model
	- Estimation: expectation maximization or SGD for neural click models
2. <span style="color:rgb(172, 96, 230)">RegressionEM</span>
	- probabilistic click model using EM require multiple observations of the same query-doc pair
	- Generalize over query-doc pairs by regressing the estimated relevance (E-step) on features
	- use <span style="color:rgb(172, 96, 230)">query-document features instead of estimating relevance separately</span> for each query-doc pair
3. <span style="color:rgb(172, 96, 230)">Neural Click Models / Two-Tower Models</span>
	- Less hand-crafting
	- <span style="color:rgb(172, 96, 230)">separate relevance (doc features) and bias (contextual features) estimation</span>
	- optimize for the likelihood of interaction data
	- Dropout/ data randomization to prevent overfitting on bias
### Advantages and Limitations of Click Models
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
Click models don't provide strong guarantees about unbiasedness, but they:

- Are relatively easy to apply
- Have lower variance than IPS
- Require fewer assumptions

Whether maximizing likelihood leads to correct estimates depends on:

- Randomization in the logged rankings
- Availability of informative features
- Having enough observations
- Whether data can be explained in multiple ways
- Whether the optimization finds the right minimum

## Advanced Estimation Methods

### The Direct Method (DM)

The <span style="color:rgb(172, 96, 230)">Direct Method uses a model trained on features to directly estimate relevance</span>:

- No selection problem (estimates relevance for every document)
- Low variance (no reweighting of interactions)
- But biased unless the regression model is perfectly correct (practically biased)
### Doubly Robust Estimation (DR)

Doubly Robust estimation combines the best of both approaches:

- Starts with the Direct Method estimate
- <span style="color:rgb(172, 96, 230)">Uses IPS to correct for biases in the DM's predictions</span>
- Unbiased if either IPS is unbiased or DM is unbiased
- Generally has lower variance than IPS alone

The DR approach <span style="color:rgb(172, 96, 230)">subtracts an unbiased IPS estimate of the DM's bias from the DM estimate</span>.

## Real-World Results and Conclusion

### Online A/B Test Results

Most research in unbiased learning to rank relies on simulation experiments that show considerable performance gains. However, real-world A/B tests typically show more modest improvements:

- The median uplift in online A/B tests is around 2.08%
- While significant, this is less dramatic than simulation results might suggest

### Ongoing and Future Research Directions

1. **Fairness in ranking**: Distributing exposure in a meritocratic manner
2. **Safety in LTR**: Ensuring new models don't decrease production performance
3. **Uncertainty and Bayesian approaches**: Moving beyond frequentist methods
4. **Other forms of bias**: Trust, item-selection, surrounding-items effects
5. **Other interaction types**: Conversion, purchases, watch-time, likes
6. **Long-term objectives**: Going beyond immediate outcomes

## Summary of Key Points

IPS, DM and DR
 Inverse-Propensity-Scoring (IPS):
- Reweights according to logging policy probabilities.
- Unbiased if propensities are correct.
- High variance due to low-propensity actions.
The Direct Method (DM):
- Purely based on regression estimates.
- Biased unless regression is 100% correct.
- Low variance by ignoring observed interactions.
Doubly Robust Estimation (DR):
- Starts with DM and uses IPS to correct it.
- Easier unbiasedness criteria.
- Lower variance than IPS if regression is decent.


- Need for learning to rank from user interactions.
- Interaction data is noisy and biased, clicks on rankings are position biased.
- Position bias can be estimated from randomized data.
- Inverse propensity scoring can unbiasedly estimate relevance from clicks.
- Click models make maximize likelihood, easy to apply, no real guarantees.
- Counterfactual LTR makes a real-world difference, but rarely of extreme degree