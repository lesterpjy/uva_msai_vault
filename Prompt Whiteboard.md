
The following text separated by dashes are abstracts of papers selected by a machine learning paper reproducibility conference. I am looking to submit to this reproducibility conference and maximizing my chances of acceptance. Analyze any possible trends from the abstracts of the following accepted papers. Potential areas of analysis may be: what topics were popular, in which ways did the reproducibility papers improve on the original papers. Give a ranked list of things that I should do for my own paper to maximize my chances of acceptance. This ranking should be based on statistics and observations you derive from these abstracts.

Graph Neural Networks have recently gained recognition for their performance on graph
machine learning tasks. The increasing attention on these models’ trustworthiness and
decision-making mechanisms has instilled interest in the exploration of explainability tech-
niques, including the model proposed in "GNNInterpreter: A probabilistic generative model-
level explanation for Graph Neural Networks." (Wang & Shen (2022)). This work aims to
reproduce the findings of the original paper, by investigation the main claims made by its
authors, namely that GNNInterpreter (i) generates faithful and realistic explanations with-
out requiring domain-specific knowledge, (ii) has the ability to work with various node and
edge features, (iii) produces explanations that are representative for the target class and
(iv) has a much lower training time compared to XGNN, the current state-of-the-art model-
level GNN explanation technique. To reproduce the results, we make use of the open-source
implementation and we test the interpreter on the same datasets and GNN models as in
the original paper. We conduct an enhanced quantitative and qualitative evaluation, and
additionally we extend the original experiments to include another real-world dataset with
larger graphs. Our results show that the first claim can only be verified for the dataset with
larger graphs. We reject the first claim for all other datasets, due to significant seed variation
and training instability, as well as extensive need for hyperparameter tuning and domain
knowledge. Furthermore, we partially validate the second claim by testing on datasets with
different node and edge features, but we reject the third claim due to GNNInterpreter’s
failure to outperform XGNN in producing dataset aligned explanations. Lastly, we are able
to confirm the last claim.

-----

Temporal graphs model complex dynamic relations that change over time, and are being
used in a growing number of applications. Recently, several graph neural networks (GNNs)
were proposed, designed specifically for this temporal setting (Xu et al., 2020; Rossi et al.,
2020). However, these models are notoriously hard to interpret. For this reason, the original
authors (Xia et al., 2023) propose the Temporal GNN Explainer (T-GNNExplainer) – an
explorer-navigator framework to efficiently compute sparse explanations of target Temporal
GNNs. We reproduce the main findings of the original paper, extend their work by proposing
a different type of navigator method, and examine in detail its explanation capabilities
and efficiency within various model and hyperparameter settings. We confirm that their
explainer outperforms the other baselines across nearly all datasets and metrics. Our findings
suggest the navigator helps bias the search process and that T-GNNExplainer can find an
exact influential event set. Moreover, we examine the effect of different navigator methods
and quantify the runtime-fidelity trade-off controlled by two hyper-parameters

-----

This paper reports on the reproducibility study on the paper ‘Explaining RL Decisions with
Trajectories’ by Deshmukh et al. (2023). The authors proposed a method to elucidate the
decisions of an offline RL agent by attributing them to clusters of trajectories encountered
during training. The original paper explored various environments and conducted a human
study to gauge real-world performance. Our objective is to validate the effectiveness of
their proposed approach. This paper conducted quantitative and qualitative experiments
across three environments: a Grid-world, an Atari video game (Seaquest), and a contin-
uous control task from MuJoCo (HalfCheetah). While the authors provided the code for
the Grid-world environment, we re-implemented it for the Seaquest and HalfCheetah en-
vironments. This work extends the original paper by including trajectory rankings within
a cluster, experimenting with alternative trajectory clustering, and expanding the human
study. The results affirm the effectiveness of the method, both in its reproduction and in the
additional experiments. However, the results of the human study suggest that the method’s
explanations are more challenging to interpret for humans in more complex environments.
Our implementations can be found on GitHub.

-----

We pursue transfer learning to improve classifier accuracy on a target task with few labeled
examples available for training. Recent work suggests that using a source task to learn a
prior distribution over neural net weights, not just an initialization, can boost target task
performance. In this study, we carefully compare transfer learning with and without source
task informed priors across 5 datasets. We find that standard transfer learning informed by
an initialization only performs far better than reported in previous comparisons. The relative
gains of methods using informative priors over standard transfer learning vary in magnitude
across datasets. For the scenario of 5-300 examples per class, we find negative or negligible
gains on 2 datasets, modest gains (between 1.5-3 points of accuracy) on 2 other datasets, and
substantial gains (>8 points) on one dataset. Among methods using informative priors, we
find that an isotropic covariance appears competitive with learned low-rank covariance matrix
while being substantially simpler to understand and tune. Further analysis suggests that the
mechanistic justification for informed priors – hypothesized improved alignment between train
and test loss landscapes – is not consistently supported due to high variability in empirical
landscapes. We release code1 to allow independent reproduction of all experiments.

----

This work aims to reproduce the findings of the paper "Fair Attribute Completion on Graph
with Missing Attributes" written by Guo, Chu, and Li [1] by investigating the claims made
in the paper. This paper suggests that the results of the original paper are reproducible
and thus, the claims hold. However, the claim that FairAC is a generic framework for many
downstream tasks is very broad and could therefore only be partially tested. Moreover,
we show that FairAC is generalizable to various datasets and sensitive attributes and show
evidence that the improvement in group fairness of the FairAC framework does not come at
the expense of individual fairness. Lastly, the codebase of FairAC has been refactored and
is now easily applicable for various datasets and models

----

To obtain state-of-the-art performance, many deeper artificial intelligence models sacrifice
human explainability in their decision-making. One solution proposed for achieving top per-
formance and retaining explainability is the Post-Hoc Concept Bottleneck Model (PCBM)
(Yuksekgonul et al., 2023), which can convert the embeddings of any deep neural network
into a set of human-interpretable concept weights. In this work, we reproduce and expand
upon the findings of Yuksekgonul et al. (2023), showing that while their claims and results
do generally hold, some of them could not be sufficiently replicated. Specifically, the claims
relating to PCBM performance preservation and its non-requirement of labeled concept
datasets were generally reproduced, whereas the one claiming its model editing capabili-
ties was not. Beyond these results, our contributions to their work include evidence that
PCBMs may work for audio classification problems, verification of the interpretability of
their methods, and updates to their code for missing implementations. The code for our
implementations can be found in https://github.com/dgcnz/FACT.

----

In this work, we attempt to reproduce the results of Enguehard (2023), which introduced
ExtremalMask, a mask-based perturbation method for explaining time series data. We
investigated the key claims of this paper, namely that (1) the model outperformed other
models in several key metrics on both synthetic and real data, and (2) the model performed
better when using the loss function of the preservation game relative to that of the deletion
game. Although discrepancies exist, our results generally support the core of the original
paper’s conclusions. Next, we interpret ExtremalMask’s outputs using new visualizations
and metrics and discuss the insights each interpretation provides. Finally, we test whether
ExtremalMask create out of distribution samples, and found the model does not exhibit this
flaw on our tested synthetic dataset. Overall, our results support and add nuance to the
original paper’s findings. Code available at this link

---

This work investigates the reproducibility of the paper " Explaining RL decisions with
trajectories “ by Deshmukh et al. (2023). The original paper introduces a novel approach in
explainable reinforcement learning based on the attribution decisions of an agent to specific
clusters of trajectories encountered during training. We verify the main claims from the
paper, which state that (i) training on less trajectories induces a lower initial state value, (ii)
trajectories in a cluster present similar high-level patterns, (iii) distant trajectories influence
the decision of an agent, and (iv) humans correctly identify the attributed trajectories
to the decision of the agent. We recover the environments used by the authors based
on the partial original code they provided for one of the environments (Grid-World), and
implemented the remaining from scratch (Seaquest and HalfCheetah, Breakout, Q*Bert).
While we confirm that (i), (ii), and (iii) partially hold, we extend on the largely qualitative
experiments from the authors by introducing a quantitative metric to further support (iii),
and new experiments and visual results for (i). Moreover, we investigate the use of different
clustering algorithms and encoder architectures to further support (ii). We could not support
(iv), given the limited extent of the original experiments. We conclude that, while some of
the claims can be supported, further investigations and experiments could be of interest.
We recognize the novelty of the work from the authors and hope that our work paves the
way for clearer and more transparent approaches

----

We evaluate CS-Shapley, a data valuation method introduced in Schoch et al. (2022) for
classification problems. We repeat the experiments in the paper, including two additional
methods, the Least Core (Yan & Procaccia, 2021) and Data Banzhaf (Wang & Jia, 2023),
a comparison not found in the literature. We include more conservative error estimates
and additional metrics, like rank stability, and a variance-corrected version of Weighted
Accuracy Drop, originally introduced in Schoch et al. (2022). We conclude that while CS-
Shapley helps in the scenarios it was originally tested in, in particular for the detection of
corrupted labels, it is outperformed by the conceptually simpler Data Banzhaf in the task
of detecting highly influential points, except for highly imbalanced multi-class problems.

----

Text-to-image generative models often present issues regarding fairness with respect to certain sensitive attributes, such as gender or skin tone. This study aims to reproduce the results presented in "ITI-GEN: Inclusive Text-to-Image Generation" by Zhang et al. (2023), which introduces a model to improve inclusiveness in these kinds of models. We show that most of the claims made by the authors about ITI-GEN hold: it improves the diversity and quality of generated images, it is scalable to different domains, it has plug-and-play capabilities, and it is efficient from a computational point of view. However, ITI-GEN sometimes uses undesired attributes as proxy features and it is unable to disentangle some pairs of (correlated) attributes such as gender and baldness. In addition, when the number of considered attributes increases, the training time grows exponentially and ITI-GEN struggles to generate inclusive images for all elements in the joint distribution. To solve these issues, we propose using Hard Prompt Search with negative prompting, a method that does not require training and that handles negation better than vanilla Hard Prompt Search. Nonetheless, Hard Prompt Search (with or without negative prompting) cannot be used for continuous attributes that are hard to express in natural language, an area where ITI-GEN excels as it is guided by images during training. Finally, we propose combining ITI-GEN and Hard Prompt Search with negative prompting.

---

This reproducibility study examines "Robust Fair Clustering: A Novel Fairness Attack and Defense Framework" by Chhabra et al. (2023), an innovative work in fair clustering algorithms. Our study focuses on validating the original paper's claims concerning the susceptibility of state-of-the-art fair clustering models to adversarial attacks and the efficacy of the proposed Consensus Fair Clustering (CFC) defence mechanism. We employ a similar experimental framework but extend our investigations by using additional datasets. Our findings confirm the original paper's claims, reinforcing the vulnerability of fair clustering models to adversarial attacks and the robustness of the CFC mechanism.

----

This paper seeks to reproduce and extend the results of the paper “Explaining Temporal Graph Models Through an Explorer-Navigator Framework” by (Xia et al., 2023). The main contribution of the original authors is a novel explainer for temporal graph networks, the Temporal GNN Explainer (T-GNNExplainer), which finds a subset of preceding events that “explain” a prediction made by a temporal graph model. The explorer is tested on two temporal graph models that are trained on two real-world and two synthetic datasets. The explorer is evaluated using a newly proposed metric for explanatory graph models. The authors compare the performance of their explorer to three baseline explainer methods, either adapted from a GNN explainer or developed by the authors. The authors claim that T-GNNExplainer achieves superior performance compared to the baselines when evaluated with their proposed metric. This work reproduces the original experiments by using the code (with minor adjustments), model specifications, and hyperparameters provided by the original authors. To evaluate the robustness of these claims, the method was extended to one new dataset (MOOC). Results show that the T-GNNexplainer performs best on some, but not all metrics as reported in the original findings. We conclude that the main lines of this paper hold up even though all results are less pronounced than claimed. Results show that the T-GNNExplainer does not perform similarly across different T-GNN models, precise dataset specifications are needed to obtain high performance, and there are simpler, less computationally costly explainer methods (like PBONE) that could offer competitive results.

---

Clustering algorithms play a pivotal role in various societal applications, where fairness is paramount to prevent adverse impacts on individuals. In this study, we revisit the robustness of fair clustering algorithms against adversarial attacks, affirming previous research findings that highlighted their susceptibility and the resilience of the Consensus Fair Clustering (CFC) model. Beyond reproducing these critical results, our work extends the original analysis by refining the codebase for enhanced experimentation, introducing additional metrics and datasets to deepen the evaluation of fairness and clustering quality, and exploring novel attack strategies, including targeted attacks on new metrics and a combined approach for balance and entropy as well as an ablation study. These contributions validate the original claims about the vulnerability and resilience of fair clustering algorithms and broaden the research landscape by offering a more comprehensive toolkit for assessing adversarial robustness in fair clustering.

---

New transformer networks have been integrated into object tracking pipelines and have demonstrated strong performance on the latest benchmarks. This paper focuses on understanding how transformer trackers behave under adversarial attacks and how different attacks perform on tracking datasets as their parameters change. We conducted a series of experiments to evaluate the effectiveness of existing adversarial attacks on object trackers with transformer and non-transformer backbones. We experimented on 7 different trackers, including 3 that are transformer-based, and 4 which leverage other architectures. These trackers are tested against 4 recent attack methods to assess their performance and robustness on VOT2022ST, UAV123 and GOT10k datasets. Our empirical study focuses on evaluating adversarial robustness of object trackers based on bounding box versus binary mask predictions, and attack methods at different levels of perturbations. Interestingly, our study found that altering the perturbation level may not significantly affect the overall object tracking results after the attack. Similarly, the sparsity and imperceptibility of the attack perturbations may remain stable against perturbation level shifts. By applying a specific attack on all transformer trackers, we show that new transformer trackers having a stronger cross-attention modeling achieve a greater adversarial robustness on tracking datasets, such as VOT2022ST and GOT10k. Our results also indicate the necessity for new attack methods to effectively tackle the latest types of transformer trackers. The codes necessary to reproduce this study are available at [https://github.com/fatemehN/ReproducibilityStudy](https://github.com/fatemehN/ReproducibilityStudy).

----

The growing reproducibility crisis in machine learning has brought forward a need for careful examination of research findings. This paper investigates the claims made by Lei et al. (2023) regarding their proposed method, LICO, for enhancing post-hoc interpretability techniques and improving image classification performance. LICO leverages natural language supervision from a vision-language model to enrich feature representations and guide the learning process. We conduct a comprehensive reproducibility study, employing (Wide) ResNets and established interpretability methods like Grad-CAM and RISE. We were mostly unable to reproduce the authors' results. In particular, we did not find that LICO consistently led to improved classification performance or improvements in quantitative and qualitative measures of interpretability. Thus, our findings highlight the importance of rigorous evaluation and transparent reporting in interpretability research.

---

Deep Learning models have taken the front stage in the AI community, yet explainability challenges hinder their widespread adoption. Time series models, in particular, lack attention in this regard. This study tries to reproduce and extend the work of Enguehard (2023b), focusing on time series explainability by incorporating learnable masks and perturbations. Enguehard (2023b) employed two methods to learn these masks and perturbations, the preservation game (yielding SOTA results) and the deletion game (with poor performance). We extend the work by revising the deletion game’s loss function, testing the robustness of the proposed method on a novel weather dataset, and visualizing the learned masks and perturbations. Despite notable discrepancies in results across many experiments, our findings demonstrate that the proposed method consistently outperforms all baselines and exhibits robust performance across datasets. However, visualizations for the preservation game reveal that the learned perturbations primarily resemble a constant zero signal, questioning the importance of learning perturbations. Nevertheless, our revised deletion game shows promise, recovering meaningful perturbations and, in certain instances, surpassing the performance of the preservation game.

---

This reproducibility study aims to evaluate the robustness of Equal Improvability (EI) - an effort-based framework for ensuring long-term fairness. To this end, we seek to analyze the three proposed EI-ensuring regularization techniques, i.e. Covariance-based, KDE-based, and Loss-based EI. Our findings largely substantiate the initial assertions, demonstrating EI’s enhanced performance over Empirical Risk Minimization (ERM) techniques on various test datasets. Furthermore, while affirming the long-term effectiveness in fairness, the study also uncovers challenges in resilience to overfitting, particularly in highly complex models. Building upon the original study, the experiments were extended to include a new dataset and multiple sensitive attributes. These additional tests further demonstrated the effec- tiveness of the EI approach, reinforcing its continued success. Our study highlights the importance of adaptable strategies in AI fairness, contributing to the ongoing discourse in this field of research.

---

In this paper, we extend the study of concept ablation within pre-trained models as introduced in 'Ablating Concepts in Text-to-Image Diffusion Models' by  (Kumari et al., 2022). Our work focuses on reproducing the results achieved by the different variants of concept ablation proposed through predefined metrics. We also introduce a novel variant of concept ablation—trademark ablation. This variant combines the principles of memorization and instance ablation to tackle the nuanced influence of proprietary or branded elements in model outputs. Further, our research contributions include an observational analysis of the model's limitations. Moreover, we investigate the model's behavior in response to ablation leakage-inducing prompts, which aim to indirectly ablate concepts, revealing insights into the model's resilience and adaptability. We also observe the model's performance degradation on images generated by concepts far from its target ablation concept, which is documented in the appendix.

---

Model guidance describes the approach of regularizing the explanations of a deep neural
network model towards highlighting the correct features to ensure that the model is “right
for the right reasons”. Rao et al. (2023) conducted an in-depth evaluation of effective and
efficient model guidance for object classification across various loss functions, attributions
methods, models, and ’guidance depths’ to study the effectiveness of different methods.
Our work aims to (1) reproduce the main results obtained by Rao et al. (2023), and (2)
propose several extensions to their research. We conclude that the major part of the original
work is reproducible, with certain minor exceptions, which we discuss in this paper. In
our extended work, we point to an issue with the Energy Pointing Game (EPG) metric
used for evaluation and propose an extension for increasing its robustness. In addition, we
observe the EPG metric’s predisposition towards favoring larger bounding boxes, a bias we
address by incorporating a corrective penalty term into the original Energy loss function.
Furthermore, we revisit the feasibility of using segmentation masks in light of the original
study’s finding that minimal annotated data can significantly boost model performance. Our
findings suggests that Energy loss inherently guides models to on-object features without the
requirement for segmentation masks. Finally, we explore the role of contextual information
in object detection and, contrary to the assumption that focusing solely on object-specific
features suffices for accurate classification, our findings suggest the importance of contextual
cues in certain scenarios.

---

In this study, we undertake a reproducibility analysis of "Learning Fair Graph Representa-
tions Via Automated Data Augmentations" by Ling et al. (2022). We assess the validity
of the original claims focused on node classification tasks and explore the performance of
the Graphair framework in link prediction tasks. Our investigation reveals that we can
partially reproduce one of the original three claims and fully substantiate the other two.
Additionally, we broaden the application of Graphair from node classification to link pre-
diction across various datasets. Our findings indicate that, while Graphair demonstrates
a comparable fairness-accuracy trade-off to baseline models for mixed dyadic-level fairness,
it has a superior trade-off for subgroup dyadic-level fairness. These findings underscore
Graphair’s potential for wider adoption in graph-based learning. Our code base can be
found on GitHub at https://github.com/juellsprott/graphair-reproducibility.


