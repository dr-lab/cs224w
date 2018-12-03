# This code calculates the average precision score, for one input parameter combination, using one output of the rev2 algorithm.

import os
import sys

NETWORKNAME = sys.argv[1]

alpha1 = int(sys.argv[2])
alpha2 = int(sys.argv[3])

beta1 = int(sys.argv[4])
beta2 = int(sys.argv[5])

gamma1 = int(sys.argv[6])
gamma2 = int(sys.argv[7])
gamma3 = int(sys.argv[8])

### LOAD THE GROUND TRUTH LABELS

f = open("./data/%s_gt.csv" % (NETWORKNAME),"r")
goodusers = set()
badusers = set()

for l in f:
        l = l.strip().split(",")
        if l[1] == "-1":
                badusers.add('u'+l[0])
        else:
                goodusers.add('u'+l[0])
f.close()
print "Number of ground truth bad users = %d, good users = %d" % (len(badusers), len(goodusers))

### COMPUTE THE ACCURACY FOR THE STORED RESULTS FROM ONE RUN OF THE REV2 ALGORITHM (USING THE GIVEN PARAMETER SETTING ONLY) ON A PARTICULAR NETWORK

fname = "./result/%s-fng-sorted-users-%d-%d-%d-%d-%d-%d-%d.csv" % (NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3)

fww = open("./result/%s-fng-sorted-users-%d-%d-%d-%d-%d-%d-%d-result.csv" % (NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3), "w")

TOTAL_COUNT = 100 # select worst 100 users

bashCommand = "wc -l %s" % fname
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
NLINES  = int(process.communicate()[0].split(" ")[0])	

bottom_precs = []
top_precs = []
X = []

for K in range(1,TOTAL_COUNT): # this loop is to calculate the precision@K score. 
	i = -1
	f = open(fname,"r")

	c11 = 0
	c12 = 0
	c21 = 0
	c22 = 0

	x = 0
	for l in f:
		i +=1
		l = l.strip().split(",")

		if i < K:
			if l[0] in goodusers:
				c11 += 1
			elif l[0] in badusers:
				c12 += 1
		elif i >= NLINES - K:
			x += 1
			if l[0] in goodusers:
				c21 +=1
			elif l[0] in badusers:
				c22 += 1
	f.close()
	#print c11, c12, c21, c22
	# store the precision at K values for both top-ranking and bottom-ranking nodes
	X.append(c21+c22+1) 
	bottom_precs.append((c22+0.001)*1.0/(c21+c22+0.001)) # adding a 0.001 score to avoid division by zero
	top_precs.append((c11+0.001)*1.0/(c11+c12+0.001))

'''
print ""
print "X = ", X
print fname.replace("-","_").split(".")[0], '=', Ys
print "pyplot.plot(X, %s, linewidth=2)" % fname.replace("-","_").split(".")[0]
'''

print "Mean average precision:\nFor fraudulent user prediction = %f\nFor benign user prediction = %f" % (numpy.mean(bottom_prec), numpy.mean(top_precs))
fww.write("%d, %d, %d, %d, %d, %d, %d, %f, %f\n" % (alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, numpy.mean(Ys), numpy.mean(Ys2)))
fww.close()

