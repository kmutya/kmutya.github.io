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
An independent set is a subset of vertices $$C \subseteq V$$ in a graph such that there is no edge between any two vertices in the independent set, $$i,j \notin E$$ for any $$i,j \in C.

 A clique (independent set) is called maximal if it is not a subset of a larger clique (independent set) in $$G$$, and maximum if there is no larger clique (independent set) in $$G$$. The cardinality of a of a maximum clique in $$G$$ is denoted $$\omega(G)$$ and is called the clique number of $$G$$. It is also known as the fractional clique number.

 Visually:

 <figure>
   <img src="{{site.url}}/images/maxclique/clique.jpg" alt="my alt text"/>
   <figcaption>image from: https://math.stackexchange.com/questions/758263/whats-maximal-clique</figcaption>
 </figure>

Given a graph $$G$$ in which we want to find a clique, we can find a complement graph $$G'$$ of $$G$$ such that for every edge $$E$$ in $$G$$ there is no edge in $$G'$$ and for every edge $$E$$ that is not in $$G$$ there is an edge in $$G'$$. **Therefore if we find an independent set in $$G'$$ it will be a clique in $$G$$.**

## Modelling
