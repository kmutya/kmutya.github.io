---
title: "Perspectives of Probability"
date: 2020-03-07
author_profile: false
mathjax: "true"
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: false
---


Based on a colleagues recommendation, I recently came across [1] which provides a great introduction in clarifying some nuances within the classical probability POV by highlighting it's connections with Measure Theory. I wanted to understand more about this measure theoretic POV of probability and compare it with classical probability theory's outlook. Here, I catalog what succedded as an outcome of that curiosity.

# Connection with Measure Theory

<figure>
  <img src="{{site.url}}/images/measuretheoryprob/1.jpg" alt="Comparision Table"/>
  <figcaption>image from [2]</figcaption>
</figure>

[2] perfectly sums up the connection between measure theory and probability theory using the above table. Now, we disect this table to enhance our understanding from both these perspectives. For convenience, we will use abbrevations: PT - for classical probability theory and MT - for measure theory treatment of probability.

We will be using [3] as reference for PT and [2] as reference for MT. For convenience, I will not be citing them over and over again, since it is understood. However, while using quotations, I will cite them explicitly. Other references will be citied explicitly.

## Sample space

### PT

* Exepriment, Sample Space: Traditionally, we use the notion of an **experiment** to define a sample space.
We consider an experiment to be non-deterministic but it's set of all outcomes to be known. This known set of all possible outcomes is defined as the **sample space.** Common notations are $S, \Omega$. Naturally, $S^c = \emptyset$.

* Events or Event Space $(E)$: Any subset of the sample space is defined as an event. If the outcome of our experiment ends up in $E$, then we can deterministically state that $E$ has occured. Now, let us think of some obvious properties that $E$ must satisy: i.) Since, $S$ is the set of all outcomes and $E$ is a subset of $S$. If there are many Event's, it should be obvious that $\bigcup_{i}E_i \in S$. ii.) ii.) Let's say that outcomes of our experiment are disjointly captured by a finite no. of events then $\bigcap_{i} E_i = \emptyset$ i.e event with no outcome. Therefore, $\emptyset \in E$. iii.) 



# References

[[1]](https://betanalpha.github.io/assets/case_studies/probability_theory.html) - Betancourt, Probability Theory

[2] - Folland, G. B. 1999. Real Analysis: Modern Techniques and Their Applications. New York: John Wiley; Sons, Inc.

[3] - Ross, S.M., 2006. A first course in probability (Vol. 7). Upper Saddle River, NJ: Pearson Prentice Hall.
