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

To aid in column generation scheme we will formulate the dual of the above equation.

$$
\begin{aligned}
	\bar{\omega}(G) = min & \sum_{I \in I^*} y_{I}\\
    s.t. \; & \sum_{I \in I_j} y_{I} \geq 1,\;\; \forall j \in V\\
\end{aligned}
$$

where $$I_j$$ denote the set of all maximal independent sets containing vertex $$j \in V$$.

Note that, for every linear programming problem there is a companion problem i.e “dual” linear program, in which the roles of variables and constraints are reversed.

In the above formulation, each variable represents a maximal independent set. Since generating the number of maximal independent sets in a graph is a NP-Hard problem as it is the same as finding the maximum clique in a compliment graph. We will use a greedy approach to generate a set of few maximal independent sets such that they cover all the vertices. The generated set $$I^\prime \subseteq I^*$$ will used as the basis (column-wise) to initialize the restricted master problem (RMP). Below is the RMP:

$$
\begin{aligned}
	\bar{\omega}(G) = min & \sum_{I \in I^\prime y_I\\
    s.t. \; & \sum_{I} \in I^\prime_j y_{I} \geq 1,\;\; \forall j \in V\\
\end{aligned}
$$

We would be implementing this post on a graph I obtained from my department. In order for it's implementation first we will build a function to read the .txt file and convert it into a list where each vertex represents another list.

```python
def preprocess(file):
  '''takes in the name of the file as a string and returns a list of edges'''
    f = open(file, 'r')
    lines = f.read().split("\n")
    col = [line.split() for line in lines] #split each line into a list
    condition = 'e' #all edges start with e
    wanted_list_3 = [i for i in col if(len(i) == 3)] #by len as some line may be empty
    wanted_list_e = [j for j in wanted_list_3 if(j[0] == condition)] #filter based on e
    wanted_list_s = [l[1:] for l in wanted_list_e] #only keep the edges
    wanted_list = [list(map(int, i)) for i in wanted_list_s] #convert string to int
    return (wanted_list)
```
In order to create the list of edges as a graph we use the networkx package in python. Traditionally we can use linked lists to represent the graph either as an adjacency list or an adjacency matrix. However, for this implementation we would use the networkx package which implements an adjacency list in the background.

```python
def create_graph(edge_list):
    '''Takes in the list of edges as input and returns a graph'''
    elist = [tuple(x) for x in edge_list] #convert sub elements to tuple as req by networkx
    G = nx.Graph()
    G.add_edges_from(elist)
    print(G.number_of_nodes())
    return (G)
```

Now we create a greedy algorithm to find a good starting basis for the RMP by generating a subset $$I^\prime$$ of maximal independent sets. We need to make sure that all the vertices of $$G$$ are included in our subset in order to serve as a good starting basis.

```python
#Main greedy algorithm
'''Takes in the graph and returns maximal independent sets'''
def greedy_init(G):
    n = G.number_of_nodes()                 #Storing total number of nodes in 'n'
    max_ind_sets = []                       #initializing a list that will store maximum independent sets
    for j in range(1, n+1):
        R = G.copy()                        #Storing a copy of the graph as a residual
        neigh = [n for n in R.neighbors(j)] #Catch all the neighbours of j
        R.remove_node(j)                    #removing the node we start from
        max_ind_sets.append([j])
        R.remove_nodes_from(neigh)          #Removing the neighbours of j
        if R.number_of_nodes() != 0:
            x = get_min_degree_vertex(R)
        while R.number_of_nodes() != 0:
            neigh2 = [m for m in R.neighbors(x)]
            R.remove_node(x)
            max_ind_sets[j-1].append(x)
            R.remove_nodes_from(neigh2)
            if R.number_of_nodes() != 0:
                x = get_min_degree_vertex(R)
    return(max_ind_sets)
```

The algorithm works as follows. For each vertex $$j$$, starting with $$j$$, all neighbours of $$j$$ are removed, and then, at each step, a minimum degree vertex is chosen from the residual graph $$R$$ and added to the maximal independent set of vertex $$j$$. Subsequently, as vertices are added to the set, their neighbours are removed and the steps are repeated until the residual graph is empty. This algorithm thus leaves us with a set of maximal independent sets for each $$j \in V$$.
