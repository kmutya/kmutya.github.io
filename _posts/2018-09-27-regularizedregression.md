---
title: "Regularized Regression"
date: 2018-09-27
header:
  image: "/images/regularizedreg/head.jpg"
mathjax: "true"
---

This post is an implementation of two common regularization techniques namely Ridge and Lasso Regression in R using a housing dataset.

When we try to fit a higher order polynomial in cases where there are a lot of independent variables we notice that our model tends to overfit (i.e have high variance). This is because, given a lot of dimensions it is not feasible to have training observations covering all different combinations of inputs. More importantly, it becomes much harder for OLS assumptions to hold ground as the no. of inputs increase. In such cases when can use regularization to control our regression coefficients and in turn reduce variance.

# Ridge Regression

Ridge Regression is particularly useful in cases where we need to retain all inputs but reduce multicolinearity and noise caused by less influential variables.

Therefore, the objective in this case is predict the target variable considering all inputs.

Mathamatically, it can be defined as:

Minimize:

$$sum_{i=1}^{n}({y_i}^2 -\hat{y}^2_i)+\lambda||w||^2_2$$

where,

Residual sum of squares = $$\sum_{i=1}^{n}({y_i}^2-\hat{y}^2_i)$$

Tuning parameter = $$\lambda$$

L2 norm = $$\||w||^2_2$$
i.e $$\sum_{i=1}^{n}w^2_i$$

Geometric intution:

The 2D contour plot below provies intution as to why the coefficients in ridge shrink to near zero but not exactly zero as we increase the tuning parameter($$\lambda$$).

<img src="{{ site.url }}{{ site.baseurl }}//images/regularizedreg/ridge.jpg" alt="Estimation picture for ridge from Giersdorf, 2017">

In the above figure RSS equation has the form of an elipse (in red) and the L2 norm for 2 variables is naturally the equation of a circle (in blue). We can observe that their interaction is when $$\beta_2$$ is close but not equal to 0.

First, let us load the packages we'd be needing.

```r
#Libraries to be used
library(glmnet) #regularized regression package
library(ggplot2) #for plotting
library(caret) #hot one encoding
library(e1071) #skewess function
```
