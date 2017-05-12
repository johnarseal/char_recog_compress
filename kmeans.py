import numpy as np



def evalDistance(vec1,vec2):
	return np.sqrt(sum(np.power(vec1-vec2,2)))


def initCents(dataSet,k):
	numS,dim = dataSet.shape

	minVal = dataSet.min()
	maxVal = dataSet.max()
	gap = float(maxVal - minVal) / (k-1)

	cents = np.zeros((k,dim))
	for i in range(k):
		cents[i,:] = [minVal + gap * i]


	return cents


def kmeans(dataSet,k):
	numSamples = dataSet.shape[0]
	clusterChanged = True
	clusterState = np.mat(np.zeros((numSamples,2)))
	centroids = initCents(dataSet,k)

	numIter = 0
	print "numSamples: " + str(numSamples)
	while clusterChanged:
		clusterChanged = False
		numChange = 0
		# for each sample
		for i in xrange(numSamples):
			minDist = 999999
			minInd = 0
			#find its cluster
			for j in range(k):
				dist = evalDistance(dataSet[i],centroids[j])
				if dist < minDist:
					minDist = dist
					minInd = j

			if int(clusterState[i,0]) != minInd:
				numChange += 1
				clusterChanged = True
				clusterState[i,:] = minInd,minDist

			if (i+1) % 100000 == 0:
				print "processed " + str(i) + " points"
	
		numIter += 1
		print "iter " + str(numIter) + ", points chanegd " + str(numChange)
	
		#update centroids
		for j in range(k):
			pointsInCluster = dataSet[np.nonzero(clusterState[:,0].A == j)[0]]
			centroids[j,:] = np.mean(pointsInCluster, axis=0)
		
		if numChange < numSamples * 0.01:
			break

	return centroids,clusterState












