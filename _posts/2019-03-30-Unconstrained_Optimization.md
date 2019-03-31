---
title: "Unconstrained Optimization"
date: 2010-03-30
header:
  image: "/images/unconstrainedopt/head.jpg"
mathjax: "true"
---

#Methods for Unconstrained Optimization

A typical unconstrained optimization problem looks like this:

$$
Min_{x \in R^n} f(x)
$$

Below are implementations of a few Unconstrained Optimization methods.

##Classical Newton Method:

Newton’s method is an algorithm for finding a zero of a nonlinear function i.e the points where a function equals 0 (minima).

The basic **idea** in Newton's method is to approximate our non-linear function $f(x)$ with a quadratic ($2^{nd}$ order approximation) and then use the minimizer of the approximated function as the starting point in the next step and repeat the process iteratively. If $f \in C^2$ function we can obtain this approximation using Taylor series expansion:

$$
f(x) = f(x_0) + (x-x_0)^T \nabla f(x) + \frac{1}{2!} (x-x_0)^T \nabla^2 f(x_0)(x-x_0)
$$

Where $x_0$ is the intitial point about which we try to approximate and $x$ is the new point. Applying FONC ($\nabla f(x) = 0$) to the above equation, we get:

$$
x = x_0 - [\nabla^2f(x)]^{-1} \nabla f(x)
$$

This $x$ is a unique global min if the Hessian ($\nabla^2f(x)$) is Positive Definite or global min if the Hessian is Positive Semi Definite. If it's not P.D we can enforce it to be P.D using LM modification. Sometimes, Newton’s method may not possess a descent property, i.e. $f (x_k+1) \geq f(x_k)$. This is especially true if the initial point $x_0$ is not sufficiently close to the solution. Therefore, we can use a stepsize ($\alpha_k$) to enforce the descent property if Hessian is P.D.

Using this stepsize Newtons method is often writen recurssively as:

$$
x_{k+1} = x_k + \alpha_k p_k \rightarrow (N.M)
$$

where $p_k$ is the solution to newton equations such as: $[\nabla^2 f(x_k)]p = -\nabla f(x_k)$. This means that the direction $p_k$ is obtained by solving a system of linear equations and not by computing the inverse of the Hessian. Step size $\alpha_k$ can be found using exact or numerical line search methods.

**Algorithm:**



**Pros:** 1. Quadratic rate of convergence due to reliance on second order information.

**Disadvantages of Newton Method:**

1. Can fail to converge or converge to a point that is not a minimum. (Can be fixed with LM modification and line search) </br>
2. Computational costs: if there are n variables, calculating the hessian involves calculating and storing $n^2$ entries and solving system of linear equations takes $O(n^3)$ operations per iteration.

To alleviate these cons, we can use methods that reduce the computational cost of Newton but at the expense of slower convergence rates. So, we make a trade-off between costs per iteration (higher for newton) and the no. of iterations (higher for these methods). These methods are based on the newton method but compute the search direction in a different manner. In newton's method the search direction is computed using

$$
p_k = - B_k^{-1} \nabla f(x_k)
$$

where $B_k = \nabla^2f(x_k)$ assuming that the Hessian is Positive Definite. Hence, in order to obtain the search direction $p_k$ we need to solve a system of linear equations.

In these methods, we instead try to approximate this $B_k$, the degree of comprise of these methods depends on the degree to which $B_k$ approximates the Hessian $\nabla^2f(x_k)$.
