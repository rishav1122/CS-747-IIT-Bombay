import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--instance",help="increase output verbosity")
parser.add_argument("--algorithm",help="increase output verbosity")
parser.add_argument("--randomSeed", type=int, help="seed")
parser.add_argument("--epsilon", type=float, help="the exponent")
parser.add_argument("--scale", type=float, help="the exponent")
parser.add_argument("--threshold", type=float, help="the exponent")
parser.add_argument("--horizon", type=int, help="seed")

args = parser.parse_args()
# print(args.x)
# 
# algoname = "kl-ucb-t1"
#
import warnings
warnings.filterwarnings("ignore")
def KL_div(p, q):
	if p==q:
		return 0
	elif q==0:
		return 0
	elif p == 1:
		return p*np.log(p/q)
	elif p == 0:
		return (1-p)*np.log((1-p)/(1-q))
	else:
		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
	
def algoklucb(meanarray,tosstimes,totaltoss):
	c=3
	limitvalue = np.log(totaltoss) + c *np.log(np.log(totaltoss))
# 	print("limit",limitva/lue)
	maxq = 0
	bandit =0
	for i in range(len(meanarray)):
		p_a = meanarray[i]
# 		print(p_a)
		if p_a==1:
# 			print("nooo")
			return i
			
		que =  np.arange(p_a, 1, 0.01)
		for q in que:
			value = tosstimes[i] * KL_div(p_a,q)
			if value > limitvalue:
# 				print("yeahhh")
				break
			tempvalue=value
			q0 = q
# 		print("value",tempvalue)
# 		print("q",q0)
		if q0>maxq:
# 			print("yess")
			maxq=q0
			bandit=i
		
# 	print("bandit",bandit)
	return bandit
	
	
def algoout(meanarray,epsilon):
	number_list = np.arange(len(meanarray))
	if(random.uniform(0, 1.0) < epsilon):
		bandit = random.choice(number_list)
	else:
		bandit = np.argmax(meanarray)
# 		print((bandit))
	return bandit

def algothom(headtimes,tosstimes,randomSeed):
	sa = headtimes
	fa = tosstimes-headtimes
	np.random.seed(randomSeed)
	samples = np.random.beta(sa+1,fa+1)
	bandit = np.argmax(samples)
# 	print((bandit))
	return bandit

def algoucb(meanarray,tosstimes,totaltoss,scale=2):
	ucbarray= meanarray + np.sqrt(scale*np.log(totaltoss)/tosstimes)
	bandit = np.argmax(ucbarray)
	return bandit
	
import random
instancename = args.instance  #instance
f = open(instancename, "r")
threshold = args.threshold
scale = args.scale
epsilon = args.epsilon #epsilon
horizon = args.horizon
rseed = args.randomSeed

import os


# 
totalTosses = horizon
# 
random.seed(rseed)
algoname = args.algorithm

if algoname == "kl-ucb-t1":
        # means = np.zeros(numCoins)
        s=[]
        for x in f:
        	s.append(float(x))
        numCoins =len(s)
        p=s
        totalHeads=0
        sumall=0
        tosses =np.zeros(numCoins)
        heads = np.zeros(numCoins)
        means = np.zeros(numCoins)
        for i in range(0, numCoins):
            coin = i 
            sumall+=p[coin]
            # coin -= 1
            outcome = "t"
            if(random.uniform(0, 1.0) < p[coin]):
                outcome = "h"
                totalHeads += 1
                heads[coin] +=1
        #                 print("outcome",outcome)
            tosses[coin] += 1
            means[coin] = heads[coin]/tosses[coin]


        for i in range(numCoins, totalTosses):



            coin = algoklucb(means,tosses,i+1)   #{would be from 0 to n-1}
            sumall+=p[coin]
            # coin -= 1
            outcome = "t"
            if(random.uniform(0, 1.0) < p[coin]):
                outcome = "h"
                totalHeads += 1
                heads[coin] +=1
            tosses[coin] += 1
            means[coin] = heads[coin]/tosses[coin]
        REW2 =max(p)*totalTosses - totalHeads
        REW2 = np.around(REW2,3)
    #         


elif algoname == "epsilon-greedy-t1":
        totalHeads = 0
        s=[]
        for x in f:
        	s.append(float(x))
        numCoins =len(s)
        p=s
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

elif algoname == "ucb-t1" or algoname == "ucb-t2":
        totalHeads = 0
        s=[]
        for x in f:
        	s.append(float(x))
        numCoins =len(s)
        p=s
        sumall=0
        tosses = np.zeros(numCoins)
        means = np.zeros(numCoins)
        heads = np.zeros(numCoins)


        for i in range(0, numCoins):
            coin = i 
            sumall+=p[coin]
            outcome = "t"
            if(random.uniform(0, 1.0) < p[coin]):
                outcome = "h"
                totalHeads += 1
                heads[coin] +=1
            tosses[coin] += 1
            means[coin] = heads[coin]/tosses[coin]

        for i in range(numCoins, totalTosses):

            coin = algoucb(means,tosses,i+1,scale)   #{would be from 0 to n-1}
            sumall+=p[coin]
            # coin -= 1
            outcome = "t"
            if(random.uniform(0, 1.0) < p[coin]):
                outcome = "h"
                totalHeads += 1
                heads[coin] +=1
            tosses[coin] += 1
            means[coin] = heads[coin]/tosses[coin]
        REW2 =max(p)*totalTosses - totalHeads
        REW2 = np.around(REW2,3) 
             

elif algoname == "thompson-sampling-t1":
        totalHeads = 0
        s=[]
        for x in f:
        	s.append(float(x))
        numCoins =len(s)
        p=s
        sumall=0
        tosses =np.zeros(numCoins)
        heads = np.zeros(numCoins)
        sumall=0
        for i in range(0, totalTosses):
            coin = algothom(heads,tosses,rseed)   #{would be from 0 to n-1}
            sumall+=p[coin]
            outcome = "t"
            if(random.uniform(0, 1.0) < p[coin]):
                outcome = "h"
                totalHeads += 1
                heads[coin] +=1
            tosses[coin] += 1
        REW2 =max(p)*totalTosses - totalHeads
        REW2 = np.around(REW2,3)

elif algoname == "alg-t3":
	firstline=True
	s=[]
	for x in f:
  		if firstline==True:
  			rewards = np.array(x.split(), dtype=np.float32)
  			firstline=False
  		else:
  			prob = np.array(x.split(), dtype=np.float32)
  			s.append(prob)
  			
	numCoins = len(s)
	p=s
	sample = random.uniform(0, 1.0)
	REW2=0
	sumall=0
	tosses =np.zeros(numCoins)
	heads = np.zeros(numCoins)
	sumall=0
	totalHeads = 0
	for i in range(0, totalTosses):
            coin = algothom(heads,tosses,rseed)   #{would be from 0 to n-1}
            sumall+=p[coin]
            sumi=0
            for i in range(len(s[coin])):
            	sumi +=s[coin][i]
            	if sample < sumi:
            		value = rewards[i]
            		totalHeads+=value
            		break
            tosses[coin] += 1
            heads[coin] +=value
            totalHeads
            
            
	REW2 =np.max(s*rewards.T)*totalTosses - totalHeads
	REW2 = np.around(REW2,3)

else:
    a=1
    REW2 = 0
# print(algoname)
        


wr2 = instancename+", "+algoname+", "+str(rseed)+", "+str(epsilon)+", "+str(scale)+", "+str(threshold)+", "+str(horizon)+", "+str(REW2)+', 0'
print(wr2)

		
	
