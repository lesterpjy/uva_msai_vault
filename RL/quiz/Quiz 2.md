---
sources:
  - "[[11 - Policy Search Methods]]"
---
> [!question] Which of the following is NOT a limitation of standard policy gradient methods, according to the text?
> a) Sensitivity to parameterization
> b) Ignoring parameter correlations
> c) Inaccurate reflection of policy behavior changes
> d) Guaranteed convergence to a global optimum
>> [!success]- Answer
>> d) Guaranteed convergence to a global optimum

> [!question] Explain the fundamental problem that NPG and TRPO aim to solve in reinforcement learning, and describe how they address this problem using the concepts of KL divergence, parameter sensitivities, and trust regions or constraints. Compare and contrast NPG and TRPO in terms of their advantages and limitations.
>> [!success]- Answer
>> The fundamental problem that NPG and TRPO aim to solve is how to make large, effective updates to a policy while maintaining stability and avoiding catastrophic performance drops.  Standard policy gradient methods suffer from sensitivity to parameterization and may make inefficient updates based on the chosen parameter space.  NPG and TRPO address this by using the KL divergence to measure policy differences directly in behavior space, which is parameterization-invariant and directly reflects the difference in the policies' actions.  They also account for parameter sensitivities and correlations using the Fisher information matrix.  NPG uses the natural gradient, which is a modified gradient direction adjusted for these sensitivities.  TRPO adds a further constraint to control the magnitude of policy updates by using a KL divergence constraint, effectively defining a trust region where approximations of the policy's return are valid.  NPG's limitation is the computational cost of inverting the Fisher information matrix.  TRPO avoids this by using a constrained optimization method, but it is still computationally intensive. While both improve on standard methods, they differ in how they manage the stability-progress tradeoff.  NPG prioritizes parameter-invariant updates, while TRPO prioritizes larger updates, albeit within a safe, constrained region. Both improve upon the standard methods by considering the policy's behavior in making updates.

> [!question] What are the key advantages of TRPO over NPG in reinforcement learning?
>> [!success]- Answer
>> TRPO offers advantages such as automatically determining step size from the KL constraint, enabling larger policy updates while maintaining stability, and demonstrating strong empirical performance on complex tasks.

> [!question] Select all the advantages of using the Kullback-Leibler (KL) divergence to measure the difference between probability distributions in policy optimization:
> a) It equals 0 when the policies are identical
> b) It is invariant to parameter transformations
> c) It naturally captures how different two policies behave
> d) It guarantees convergence to an optimal solution
> e) It simplifies the computation of the Fisher information matrix
>> [!success]- Answer
>> a) It equals 0 when the policies are identical
>> b) It is invariant to parameter transformations
>> c) It naturally captures how different two policies behave

> [!question] The Fisher information matrix, represented as $F$, is calculated as the `____` of the `____` of the log-policy gradient.
>> [!success]- Answer
>> expectation, outer product

> [!question] The Natural Policy Gradient (NPG) and Trust Region Policy Optimization (TRPO) methods are primarily concerned with finding the direction of steepest improvement in parameter space.
>> [!success]- Answer
>> False

