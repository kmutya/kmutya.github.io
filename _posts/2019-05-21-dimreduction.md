---
title: "Dimensionality Reduction"
date: 2019-05-19
header:
  image:
mathjax: "true"
---

This post focuses on the mathematical underpinnings of dimensionality reduction and raw numpy implementations.

## Basic Setting

**Setting:** Have a dataset $$D \in R^{n*d}$$ i.e

$$D = \begin{bmatrix}x_{11} & x_{12} & \dots & x_{1d}\\\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \dots & x_{nd} \end{bmatrix}$$

where each row is a vector in a $$d$$-dimensional cartesian coordinate space, $$x_i \in R^d$$ i.e $$x_i = (x_{i1},x_{i2},x_{i3},...x_{id})^T \forall i \in \{1...n\}$$ and each row is spanned by '$$d$$' standard basis vectors i.e $$e_1,e_2...e_d$$ where $$e_j$$ corresponds to the jth predictor $$\in D$$ i.e the jth standard basis vector $$e_j$$ is the d-dimensional unit vector whose jth component is 1 and the rest of the components are 0, $$e_j = (0,...1,...0)$$.


Since it is a standard basis, we observe two interesting properties:

$$
i.) \ e_i^T.e_j = 0  \ \forall \ i,j \ni i\neq j \\
ii.) \ ||e_i|| = 1
$$

Now, given any other set of $$d$$-orthonormal vectors $$\{u_1,u_2,...,u_d\}$$ we can express each row $$x_i \forall \ i \in \{1,2,...,n\}$$, which can also be thought of as a point in a $$d$$-dimensional space, as a linear combination of the new $$d$$-orthonormal vectors and some constant vector $$a = (a_1,a_2,...,a_d)^T$$. In matrix form,

$$x = Ua$$ where $$U \in R^{d*d}, a \in R^d$$.

Note that, U is a diagonal matrix since it comprises of d-dimensional unit vectors which have a value of 1 corresponding to the jth predictor and 0 otherwise.

**Key Questions:** There are some natural questions to ask in regards to the above setting:

1. As there are infinite choices for the set of orthonormal basis vectors, can we somehow select an optimal basis?

2. If $$d$$ is very large can we find a subspace of $$d$$ such that essential charecterstics of data are still preserved?

Asking these questions stem the notion of dimensionality reduction.

Formally, the objective with dimensionality reduction is to seek an $$r$$ dimensional basis $$\ni r << d$$ that gives the best approximation of the projection of all points $$x_i \in D$$ on $$x_i' \in D'$$ where $$D'$$ is the r-dimensional subspace of D. This can alternatively be viewed as minimizing $$\epsilon = x_i-x_i'$$ over all $$x_i \forall i \in \{1,2...,n\}$$

## Principal Component Analysis (PCA)

**Idea:** The basic idea is to project points from our Dataset $$D$$ onto a lower dimensional subspace $$D'$$ in such a way that we still capture most of the variation in our data.

### 1. Project points from D onto the subspace with a basis denoted by U.

$$
x_i = Proj_{subspace}x_i + Proj_{subspace}^{\perp}x_i \\
x_i = C(U) + C(U)^{\perp} = C(U) + N(U^{\perp}) = C(U) + 0 \\
\implies x_i = Ua
$$

This makes intuitive sense sense as any point in the projected subspace should be reachable as a linear combination of the basis vectors and some constants.

We can solve for a as follows,

$$
x_i - Ua = 0 \\
U^Tx_i - U^TUa = 0 \\
\implies a = U^Tx_i(U^TU)^{-1} \\
\implies a = U^Tx_i
$$

The above 'a' gives coordinates of projected points in the new basis.
Note that the above inverse always exists as U being a orthogonal matrix is linearly independent and non-singular. Furthermore, since U is orthogonal,
$$
U^{-1} = U^T \implies U^TU = I
$$

### 2. Looking at the variance of projected points along the subspace

$$
\sigma^2_u = \frac{1}{n} \sum_{i=1}^n(a_i - \mu_u)^2 \\
$$
$$\mu_u = 0$$ if we center the data.
$$
\sigma^2_u = \frac{1}{n} \sum_{i=1}^n(a_i)^2 \\
= \frac{1}{n} \sum_{i=1}^n(U^Tx_i)^2 \\
= \frac{1}{n} \sum_{i=1}^n(U^T(x_ix_i^T)U) \\
= U^T (\frac{1}{n} \sum_{i=1}^n x_ix_i^T) U \\
= U^T\Sigma U
$$

where $$\Sigma$$ is the covariance matrix of the centered data.

### 3. (An answer to the first key question): One way to select an optimum basis among all basis would be to choose a basis that maximizes the projected variance.

The above statement is super intuitive. Since, we want to find a low dimensional representation of our dataset it makes sense to choose a basis that maximizes this projected variance.

Assuming a 1-D subspace. We can write out the equality constrained optimization problem as follows:

$$
Max \ U^T\Sigma U \\
s.t \ U^TU = 1 \\
$$

Using Langrange multipliers we can rewrite this constrained optimization problem into an unconstrained one and solve it.

$$
Max \ U^T\Sigma U - \lambda(U^TU - 1) \\
\frac{\partial U^T\Sigma U - \lambda(U^TU - 1)}{\partial U} = 0 \\
2\Sigma U - 2\lambda U = 0 \\
\implies \Sigma U = \lambda U
$$
