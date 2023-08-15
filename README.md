# eigen-sens
Eigenvalue and eigenvector derivative examples


## Theory
The repo contains code that implements two formulas (one using the adjoint method and the other using reverse algorithmic differentiation (RAD)) to efficiently compute **the derivative of the functionals of the form $f(\lambda(x), \phi(x))$** where $\lambda$ and $\phi$ is one eigenvalue-eigenvector pair (e.g., the eigenvalue with largest real part or magnitude together with its corresponding eigenvector) and $x$ is the design variables that parametrize the eigenvalue problem.

The adjoint method is more general but is a bit more complex.
The RAD formula is more elegant but specific.
For an eigenvalue problem that takes the following form:

> $\mathbf{A}\boldsymbol{\phi} = \lambda \boldsymbol{\phi}$

And the corresponding left eigenproblem is formed to be used later.

> $\mathbf{A}^*\tilde{\boldsymbol{\phi}} = \lambda^* \tilde{\boldsymbol{\phi}}$


The derivative of the eigenvalue, $\lambda$ with respect to the entries of the coefficient matrix, $\mathbf{A}$ takes the following form (decomposed into real and imaginary parts)

> $
\begin{aligned}
\frac{\mathrm{d} \lambda_r}{\mathrm{d} \mathbf{A}_r} &= \mathrm{Re}\left(\frac{\tilde{\boldsymbol{\phi}} {\boldsymbol{\phi}}^*}{\boldsymbol{\phi}^*\tilde{\boldsymbol{\phi}}}\right),\\
\frac{\mathrm{d} \lambda_r}{\mathrm{d} \mathbf{A}_i} &= \mathrm{Im}\left(\frac{\tilde{\boldsymbol{\phi}} {\boldsymbol{\phi}}^*}{\boldsymbol{\phi}^*\tilde{\boldsymbol{\phi}}}\right),\\
\frac{\mathrm{d} \lambda_i}{\mathrm{d} \mathbf{A}_r} &= -\frac{\mathrm{d} \lambda_r}{\mathrm{d} \mathbf{A}_i},\\
\frac{\mathrm{d} \lambda_i}{\mathrm{d} \mathbf{A}_i} &= \frac{\mathrm{d} \lambda_r}{\mathrm{d} \mathbf{A}_r}.
\end{aligned}
$

It is assumed that the eigenvalue is an analytic function of the coefficient matrix (This is in general true. But there are corner cases that this does not hold, e.g., repeated eigenvalues).
Use the [link](https://www.researchgate.net/profile/Sicheng-He/publication/362931690_Eigenvalue_problem_derivatives_computation_for_a_complex_matrix_using_the_adjoint_method/links/6362785654eb5f547c993819/Eigenvalue-problem-derivatives-computation-for-a-complex-matrix-using-the-adjoint-method.pdf) for more details or check the bottom of the description for more details. 

## There are so many formulas for eigenvalue-eigenvector sensitivity. Why another formula?
In the literature, most formulas are "forward" methods meaning the derivative is propagated in the forward direction of the computation graph. 
The method's computational cost is proportional to the number of design variables, $\mathit{O}(n_x)$.
This is inefficient if there is a large number of design variables.
**We derive the adjoint and RAD formulas to achieve $\mathit{O}(1)$ scaling.**

> **_NOTE:_** There is another classic adjoint method for eigenvalue-eigenvector derivative computation in the literature by [Murthy and Haftka](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620260202) but this method has a computational cost that scales with $\mathit{O}(n_x)$. 
It was named after the adjugate matrix (or the classic adjoint matrix).

For more details about derivative (sensitivity, gradient) computation, read the Chapter 6 of [mdobook](http://flowlab.groups.et.byu.net/mdobook-dark.pdf).

## Reference
This repo contains the code of several eigenvalue and eigenvector derivative formulas derived in the following publication:

> Sicheng He, Yayun Shi, Eirikur Jonsson, and Joaquim R.R.A. Martins. Eigenvalue problem derivatives computation for a complex matrix using the adjoint method. Mechanical Systems and Signal Processing, 185:109717, 2023. doi:10.1016/j.ymssp.2022.109717

The paper is also available to download via the [link](https://www.researchgate.net/profile/Sicheng-He/publication/362931690_Eigenvalue_problem_derivatives_computation_for_a_complex_matrix_using_the_adjoint_method/links/6362785654eb5f547c993819/Eigenvalue-problem-derivatives-computation-for-a-complex-matrix-using-the-adjoint-method.pdf). 
