import numpy as np

file1 = open("outputklucb1.txt", "w+")
file2 = open("outputklucb2.txt", "w+")

algoname = "kl-ucb-t1"

def KL_div(p, q):
	if p==q:
		return 0
	if q==0:
		return 0
	if p == 1:
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

epsilon = 0.02
horizons = [100, 400, 1600, 6400, 25600, 102400]
names = ["instances/instances-task1/i-1.txt","instances/instances-task1/i-2.txt","instances/instances-task1/i-3.txt"]
for horizon in horizons:
    for instancename in names:
        for rseed in range(50):
            import random
            import os
            f = open(instancename, "r")
            # print((f.readline())
            s=[]
            for x in f:
#               print((x))
              s.append(float(x))
    
            # 

            numCoins = len(s)
            # numCoins = 3
            # p = [0.6, 0.3, 0.8]
            p=s

            totalTosses = horizon

            random.seed(rseed)

            totalHeads = 0


            sumall=0
            tosses =np.zeros(numCoins)
            

            means = np.zeros(numCoins)
            heads = np.zeros(numCoins)



            for i in range(0, numCoins):

            #     os.system('clear')

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
#                 tossesFinished += 1
#                 print("coin",coin)





            for i in range(numCoins, totalTosses):



                coin = algoklucb(means,tosses,i+1)   #{would be from 0 to n-1}
                sumall+=p[coin]
                # coin -= 1
                outcome = "t"
                if(random.uniform(0, 1.0) < p[coin]):
                    outcome = "h"
                    totalHeads += 1
                    heads[coin] +=1
#                 print("outcome",outcome)
#                 outcomes[coin][tosses[coin]] = outcome
                tosses[coin] += 1
                means[coin] = heads[coin]/tosses[coin]
#                 tossesFinished += 1
#                 print("coin",coin)


            # print( "Total heads = " + str(totalHeads))
            REW1 = max(p)*totalTosses - sumall
            REW1 = np.around(REW1,3)
            REW2 =max(p)*totalTosses - totalHeads
#             REW2 = np.around(REW,3)
            # print(REW)
            wr1 = "../"+instancename+", "+algoname+", "+str(rseed)+', 0.02'+', 2'+', 0'+", "+str(horizon)+", "+str(REW1)+', 0'+"\n"
            file1.write(wr1)
#             print(wr1)
            wr2 = "../"+instancename+", "+algoname+", "+str(rseed)+', 0.02'+', 2'+', 0'+", "+str(horizon)+", "+str(REW2)+', 0'+"\n"
            file2.write(wr2)
#             print(wr2)s

		
	
