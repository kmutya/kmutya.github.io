---
title: "Dimensionality Reduction"
date: 2019-05-19
header:
  image:
mathjax: "true"
---

This post focuses on the mathematical underpinnings of dimensionality reduction and raw numpy implementations.

#Basic Setting

**Setting:** Have a dataset $D \in R^{n*d}$ i.e

$D = \begin{bmatrix}x_{11} & x_{12} & \dots & x_{1d}\\\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \dots & x_{nd} \end{bmatrix}$

where each row is a vector in a $d$-dimensional cartesian coordinate space, $r_i \in R^d$ i.e $r_i = (x_{i1},x_{i2},x_{i3},...x_{id})^T \forall i \in \{1...n\}$ and each row is spanned by '$d$' standard basis vectors i.e $e_1,e_2...e_d$ where $e_j$ corresponds to the jth predictor $\in D$ i.e the jth standard basis vector $e_j$ is the d-dimensional unit vector whose jth component is 1 and the rest of the components are 0, $e_j = (0,...1,...0)$.


Since it is a standard basis, we observe two interesting properties: </br>
i.) $e_i^T.e_j = 0  \ \forall \ i,j \ni i\neq j$ </br>
ii.) $||e_i|| = 1$

Now, given any other set of $d$-orthonormal vectors $\{u_1,u_2,...,u_d\}$ we can express each row $r_i \forall \ i \in \{1,2,...,n\}$, which can also be thought of as a point in a $d$-dimensional space, as a linear combination of the new $d$-orthonormal vectors and some constant vector $a = (a_1,a_2,...,a_d)^T$. In matrix form,

$r = Ua$ where $U \in R^{d*d}, a \in R^d$.

Note that, U is a diagonal matrix since it comprises of d-dimensional unit vectors which have a value of 1 corresponding to the jth predictor.







#Principal Component Analysis (PCA)
