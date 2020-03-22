import numpy as np
import matplotlib.pyplot as plt

#Example 1.
#Measureable Space X
x = np.random.poisson(5, 1000)
plt.hist(x)

#Function 
def fun(input):
    y = np.sqrt(x)
    return(y)

#Transformation on Y
y = np.apply_along_axis(fun, 0, x)
plt.hist(y)

#Example 2. 
#Simulate a coin toss
ct = []
for i in range(100):
    ct.append(np.random.randint(2,size = 100))
plt.hist(ct) #Let head's be 1.

#Random variable is sum of all Heads
sh = []
for i in range(100):
    sh.append(np.sum(ct[i]))
plt.hist(sh)

#Example 3.
x = []
for i in range(20):
    x.append(np.random.poisson(5, 1000))
plt.hist(x)

#Function 
def fun(input):
    y = (input*3) - 4
    return(y)

#Transformation on Y
y = np.apply_along_axis(fun, 0, x)
plt.hist(y)


