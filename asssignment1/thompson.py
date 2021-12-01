import numpy as np


def algothom(headtimes,tosstimes):
	sa = headtimes
	fa = tosstimes-headtimes
	np.random.seed(randomSeed)
	samples = np.random.beta(sa+1,fa+1)
	bandit = np.argmax(samples)
	print((bandit))
	return bandit

epsilon = 0.4

f = open("cs747-pa1-v1/instances/instances-task1/i-3.txt", "r")
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

totalTosses = 50
randomSeed = 1

random.seed(randomSeed)

totalHeads = 0



outcomes = []
tosses = []
for c in range (0, numCoins):
    tosses.append(0)
    o = []
    for i in range(0, totalTosses):
        o.append("-")
    outcomes.append(o)
tossesFinished = 0

means = np.zeros(numCoins)
heads = np.zeros(numCoins)
sumall=0
for i in range(0, totalTosses):

    os.system('clear')
    for c in range(0, numCoins):
        s = ""
        for i in range(0, tosses[c]):
            s = s + outcomes[c][i] + " "
        print( "Coin " + str(c + 1) + ": " + s)
    # print()
#     print( "Tosses left = " + str(totalTosses - tossesFinished))
#     print()
#     print( "------------------------------------")
	
    coin = algothom(heads,tosses)   #{would be from 0 to n-1}
    # coin -= 1
    sumall+=p[coin]
    outcome = "t"
    if(random.uniform(0, 1.0) < p[coin]):
        outcome = "h"
        totalHeads += 1
        heads[coin] +=1
#     print("outcome",outcome)
    outcomes[coin][tosses[coin]] = outcome
    tosses[coin] += 1
    means[coin] = heads[coin]/tosses[coin]
    tossesFinished += 1
#     print("coin",coin)
    
#     a=input()
    
print("jeads",heads)
print("num",tosses)
# os.system('clear')
print()
for c in range(0, numCoins):
    s = ""
    for i in range(0, tosses[c]):
        s = s + outcomes[c][i] + " "
    print( "Coin " + str(c + 1) + ": " + s)

print( "Total heads = " + str(totalHeads))
REW = max(p)*totalTosses - sumall
print(REW)


		
	
