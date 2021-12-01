import numpy as np

f1 = open("outputepsilon1.txt", "w+")
f2 = open("outputepsilon2.txt", "w+")


def algoout(meanarray,epsilon = 0.02):
	number_list = np.arange(len(meanarray))
	if(random.uniform(0, 1.0) < epsilon):
		bandit = random.choice(number_list)
# 		print((bandit))
	else:
		bandit = np.argmax(meanarray)
# 		print((bandit))
	return bandit

epsilon = 0.4

f = open("instancename, "r")
# print((f.readline())
s=[]
for x in f:
  print((x))
  s.append(float(x))
# print((len(x))
# print((s)
# print((len(s))
#!/usr/bin/python
# 
import random
import os

numCoins = len(s)
# numCoins = 3
# p = [0.6, 0.3, 0.8]
p=s

totalTosses = 500
randomSeed = 1

random.seed(randomSeed)

totalHeads = 0



tosses = np.zeros(numCoins)


means = np.zeros(numCoins)
heads = np.zeros(numCoins)
sumall=0
for i in range(0, totalTosses):
	
    coin = algoout(means,epsilon)   #{would be from 0 to n-1}
    # coin -= 1
    sumall+=p[coin]
    outcome = "t"
    if(random.uniform(0, 1.0) < p[coin]):
        outcome = "h"
        totalHeads += 1
        heads[coin] +=1
        sumall+=1
    tosses[coin] += 1
    means[coin] = heads[coin]/tosses[coin]

REW1 = max(p)*totalTosses - sumall
REW2 =max(p)*totalTosses - totalHeads
# print(REW)
print("../", instance)




		
	
