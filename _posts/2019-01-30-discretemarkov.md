---
title: "Ranking teams using Discrete Time Markov Chains"
date: 2019-01-30
tags: [Markov Chains, Stochastic Modelling]
header:
  image: "/images/discretemarkov/cover.jpg"
mathjax: "true"
---

# Background

Ranking methods are an essential tool in making decisions. They have many applications from sports to web searches to recommender systems. One of the most popular ranking algorithm is Google's Page Rank algorithm that also uses Markov Chains in some capacity. In this post we use Discrete Time Markov Chains (DTMC's) to rank all the 32 NFL teams after the regular season.
The National Football League (NFL) is a professional American football league consisting of 32 teams, divided equally between the National Football Conference (NFC) and the American Football Conference (AFC). Both conferences consist of four four-team divisions. Each team plays 16 regular-season games; thus, teams do not play all other teams during a single regular season.
We will be using scores from the 2007 regular season, can be downloaded from: [link](https://www.pro-football-reference.com/years/2007/games.html)

## Modelling the problem

Naturally, we will consider each team to correspond to each state in the Markov chain.

Therefore, Our state space will be $X_n$ i.e the total no. of football teams in NFL.

Therefore, $$X_n$$ $$\in$$ {0,1,2...31}

We introduce a new paramter $$F$$ i.e Football fans where $$f$$ is an individual fan.

Initially, we assume that football fans are equally distributed among all the teams i.e $$F_0 = F_1 = F_2 = ... = F_{31}$$

Then,

</br>

$$\exists$$ $$P_{i,j}$$ $$\forall$$ $$i = j$$ i.e Transition exists for all the teams that have played each other

$$\nexists$$ $$P_{i,j}$$ $$\forall$$ $$i \ne j$$ i.e Transition does not exist for all the teams that have not played each other

</br>

Based on the values of scores (subjective to the approach used), we assume that after each match a fan makes a decision to move i.e switch allegiance from one team to another. Eventually, we consider the team with the highest no. of fans to be ranked 1.

</br>
Note that, *movement of fans* here is an arbitirary concept and it is the same as saying *chances of winning of the Vince Lombardi Trophy* $$t$$, where after each match $$t$$ moves from team $$i \longrightarrow j$$ or $$j \longrightarrow i$$. So, in conclusion here we are simply equating $$F$$ (No. of fans) to chances of winning the trophy.

With this construct we can say that our problem is a first order **Markov Chain** as $P(X_{n+1} = j| X_n = i )$ and $P(X_{n+1} = i| X_n = j)$.

## Approach 1 : Using both PtsW and PtsL

In this approach at each iteration,

\[
 f_{i,k} \longrightarrow f_{j,k+1}
\]

1. A fan moves from losing team to winning team based on PtsW i.e points scored by the winning team

\[
 f_{j,k} \longrightarrow f_{i,k+1}
\]

2. A fan moves from the winning team to losing team based on PtsL i.e points scored by the losing team

Note that, in this case a fan does not move from the same team to itself i.e $f_{i,k} \not\to f_{i,k+1}$ as we are explicitly using scores and there is naturally no PtsW and PtsL data for a team against itself. Therefore, by extension: $\nexists$ $P_{i,i}$ $\forall$ $i$

Our Markov Chain is also **irreducible** simply by how the league is scheduled^[1]. To elaborate, in an NFL league each team plays 16 games each season.

* Twice agianst each team in their division (6)
* 4 games against a division in the other league (4)
* 4 games against another division in their league (4)
* 1 game each against two of the remaining divisions in their league (2)

So a team is connected to all the other 32 teams.

For **aperiodicity**, first consider one division where all 4 teams play each other twice. We will have 4 bi-directed states that are all connected to each other. Note that a Markov chain is **aperiodic** if there are 3 or more fully connected bidirected states. By extension, our division is aperiodic. Now as our chain is irreducible and since aperiodicity is a class property, our entire model is also aperiodic.

Now, that we have proved that our markov chain is **irreducible and aperiodic** we can use the property of $\pi$ = $\pi P$ to obtain the steady state vector $\pi$ which can then be used to rank the teams.

Post some code this is what our matrix looks like with only PtsW and PtsL values.



![Initial Matrix after filling in Ptsw and PtsL](/Users/apple/Desktop/assignment/ap1_initial.png) ![Matrix with probabilites where rows now add up to 1](/Users/apple/Desktop/assignment/ap1_2.png)

Note that, in cases where there were two matches between teams we have taken the sum of Ptsw and PtsL.

\newpage
After ranking the steady state $\pi$ vector this is what we get:

1. New England Patriots
2. Indianapolis Colts
3. Green Bay Packers
4. Dallas Cowboys
5. San Diego Chargers
6. New York Giants
7. Chicago Bears
8. Detroit Lions
9. Jacksonville Jaguars
10. Minnesota Vikings
11. Houston Texans
12. Philadelphia Eagles
13. New Orleans Saints
14. Cincinnati Bengals
15. Cleveland Browns
16. Pittsburgh Steelers
17. Denver Broncos
18. Arizona Cardinals
19. Miami Dolphins
20. Washington Redskins
21. New York Jets
22. Seattle Seahawks
23. Buffalo Bills
24. Tampa Bay Buccaneers
25. Baltimore Ravens
26. Tennessee Titans
27. Oakland Raiders
28. St. Louis Rams
29. Atlanta Falcons
30. Kansas City Chiefs
31. San Francisco 49ers
32. Carolina Panthers

##Approach 2 : Using absolute wins and losses

The same construct that we defined in the Approach 1 carries over to this one with the only difference being that instead of PtsW and PtsL we consider wins and losses of a team.

At each iteration,

\[
 f_{i,k} \longrightarrow f_{j,k+1}
\]

1. A fan moves from the losing team to the winning team with a probability $p \in (0.5, 1)$

\[
 f_{j,k} \longrightarrow f_{i,k+1}
\]

2. A fan moves from the winning team to losing team with probability $1-p$

Note that, the probability of a fan moving to a winning team should be higher than a losing team. Mathamatically, $P(f_{i_w})>P(f_{j_l})$

Secondly, if two teams have equal wins and loses against each other then the transitions between the two teams are considered to be equally likely in either direction i.e $f_{i,j} = f_{j,i} = 0.5$.

The chain is irreducible and aperiodic due to the same reasons as mentioned in Approach 1.

In this case we choose an arbitirary p value of $0.8$


![Matrix with initial probabilites of p and (1-p)](/Users/apple/Desktop/assignment/ap2_1.png) ![Matrix after normalization](/Users/apple/Desktop/assignment/ap2_2.png)

\newpage

Using $\pi = \pi P$ and ranking the steady state $\pi$ vector this is what we get:

1. New England Patriots
2. Dallas Cowboys
3. Green Bay Packers
4. San Diego Chargers
5. Indianapolis Colts
6. Washington Redskins
7. Jacksonville Jaguars
8. Philadelphia Eagles
9. New York Giants
10. Chicago Bears
11. Minnesota Vikings
12. Tennessee Titans
13. Denver Broncos
14. Houston Texans
15. Detroit Lions
16. Cleveland Browns
17. Pittsburgh Steelers
18. Buffalo Bills
19. Tampa Bay Buccaneers
20. New Orleans Saints
21. Seattle Seahawks
22. Carolina Panthers
23. Arizona Cardinals
24. Cincinnati Bengals
25. Kansas City Chiefs
26. Baltimore Ravens
27. New York Jets
28. Oakland Raiders
29. Atlanta Falcons
30. Miami Dolphins
31. San Francisco 49ers
32. St. Louis Rams

##Approach 3 : Difference of PtsW-PtsL

In this case we consider only the difference of points scored i.e (PtsW - PtsL) between 2 teams to indicate that the fan of a losing team moves from it to a winning team.

\[
 f_{i,k} \longrightarrow f_{j,k+1}
\]


Additionally, as New England Patriots have zero losses we assign a probability of $1/32$ for a fan to move from Patriots to any other team in our state space or stay with Patriots. Intutively, it means that as Patriots are undefeated all the time Patriot fans have equal probability of shifting to any other team or remain with Patriots if they choose to do so after each match. By doing so we relax our assumption, only for Patriots, that a fan does not move from the same team to itself.

By the nature of league scheduling, each team is connected to all the other teams in our state space, even though we do not have bidirected states like in approach 1 or 2. Our Markov Chain is **irreducible** as each team has won at least once, hence there exists a probability to come back to the team in finite steps.

Moreover, as Patriots division has 4 bi-directed fully connected states. The division is **aperiodic** and by extension our chain is **aperiodic**.


![Matrix with initial probabilites of p and (1-p)](/Users/apple/Desktop/assignment/ap3_1.png) ![Matrix after normalization](/Users/apple/Desktop/assignment/ap3_2.png)

\newpage
Using $\pi = \pi P$ and ranking the steady state $\pi$ vector this is what we get:

1. New England Patriots
2. Dallas Cowboys
3. Indianapolis Colts
4. Washington Redskins
5. Green Bay Packers
6. Chicago Bears
7. San Diego Chargers
8. Tennessee Titans
9. Jacksonville Jaguars
10. Pittsburgh Steelers
11. Minnesota Vikings
12. New Orleans Saints
13. Houston Texans
14. Philadelphia Eagles
15. Seattle Seahawks
16. Cincinnati Bengals
17. Detroit Lions
18. New York Giants
19. Denver Broncos
20. Tampa Bay Buccaneers
21. Arizona Cardinals
22. Kansas City Chiefs
23. Carolina Panthers
24. Cleveland Browns
25. Baltimore Ravens
26. Buffalo Bills
27. Atlanta Falcons
28. San Francisco 49ers
29. New York Jets
30. Oakland Raiders
31. St. Louis Rams
32. Miami Dolphins

#References

1. NFL league scheduling: https://www.youtube.com/watch?v=KGKwTnaV-rg
2. Data from https://www.pro-football-reference.com/years/2007/games.htm#
3. Background reading: Vaziri, B. (n.d.). Markov-based ranking methods
