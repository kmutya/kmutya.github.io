---
title: "Maximum Clique Problem: Linear Programming Approach"
date: 2019-02-06
tags: [Linear Programming]
header:
  image: "/images/maxclique/cover.jpg"
mathjax: "true"
---


In this article we look at the NP-hard maximum clique problem and solve it using a Linear Programming approach.
In particular, we reduce the clique problem to an Independent set problem and solve it by appying linear relaxation and column generation. MCP was one of the 21 original NP-hard problems Karp enumerated in 1972.

## Background

The clique problem arises in many real world settings. Most commonly in social networks which can be represented as a graph where the Vertices represent people and the graph's edges represent connections. Then a clique represents a subset of people who are all connected to each other. So by using clique related algorithms we can find more information regarding the connections. Along with this, the clique problem also has many applications in computer vision and pattern recognition, patterns in telecommunications traffic, bioinformatics and computational chemistry.

An undirected graph has vertices that are connected together by undirected edges which we'll represent as $$G = (V, E)$$ where $$V = {1,2, . . , n}$$ and $$E \subseteq V \times V$$.
A clique is a subset of vertices $$C \subseteq V$$ in a graph such that there is an edge between any two vertices in the clique, $$i,j \in E$$ for any $$i,j \in C$$.
An independent set is a subset of vertices $$C \subseteq V$$ in a graph such that there is no edge between any two vertices in the independent set, $$i,j \notin E$$ for any $$i,j \in C$$.

 A clique (independent set) is called maximal if it is not a subset of a larger clique (independent set) in $$G$$, and maximum if there is no larger clique (independent set) in $$G$$. The cardinality of a of a maximum clique in $$G$$ is denoted $$\omega(G)$$ and is called the clique number of $$G$$. It is also known as the fractional clique number.

 Visually:

 <figure>
   <img src="{{site.url}}/images/maxclique/clique.jpg" alt="my alt text"/>
   <figcaption>image from: https://math.stackexchange.com/questions/758263/whats-maximal-clique</figcaption>
 </figure>

Given a graph $$G$$ in which we want to find a clique, we can find a complement graph $$G'$$ of $$G$$ such that for every edge $$E$$ in $$G$$ there is no edge in $$G'$$ and for every edge $$E$$ that is not in $$G$$ there is an edge in $$G'$$. **Now, if we find an independent set in $$G'$$ it will be a clique in $$G$$.**

## Modelling

Let $$I^*$$ denote the set of all maximal independent sets in $$G$$. Then the maximum clique problem can be formulated as the following integer program:

$$
\begin{aligned}
maximize & \; \sum_{j \in V} x_j\\
subject\;to\;  &   \sum_{j \in V} x_j \leq 1, \forall I \in I^*\\
&  x_j \in \{0,1\}, \; j \in V.
\end{aligned}
$$

In the above equation the decision variable $$x_j \forall j \in V$$ takes a biniary value. 1 if it is included in the maximum clique, 0 otherwise. The constraint of the above LP is has an upper bound of 1 as every maximal independent set can atmost have one vertex member of the set belonging to the maximum clique if there are more than 1 vertex in a maximal clique then that set is not independent. Moreover, note that constraint corresponds to every maximal independent set $$I \in I^*$$. We now relax the model by replacing the integer constraints with non-negativity. We obtain the following linear program yielding an upper bound $$\bar{\omega} \geq \omega(G)$$:

$$
\begin{aligned}
\bar{\omega}(G) = max & \sum_{j \in V} x_j\\
	s.t. \; & \sum_{j \in V} x_j \leq 1,\;\; I \in I'\\
	&  x_j \geq 0, \; j \in V.
\end{aligned}
$$

Now we use duality to make explicit the effect of changes in the constraints on the value of the objective and also aid in the column generation scheme. Below is the dual of the above equation.

$$
\begin{aligned}
	\bar{\omega}(G) = min & \sum_{I \in I^*} y_{I}\\
    s.t. \; & \sum_{I \in I_j} y_{I} \geq 1,\;\; \forall j \in V\\
	&  x_j \geq 0, \; I \in I^*,
\begin{aligned}
$$

where $$I_j$$ denote the set of all maximal independent sets containing vertex $$j $\in$ V$$.

Note that, for every linear programming problem there is a companion problem i.e “dual” linear program, in which the roles of variables and constraints are reversed.
