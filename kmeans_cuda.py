import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def evalDistance(vec1,vec2):
	return np.sqrt(sum(np.power(vec1-vec2,2)))


def initCents(dataSet,k):
	dim = 1

	minVal = dataSet.min()
	maxVal = dataSet.max()
	gap = float(maxVal - minVal) / (k-1)

	cents = np.zeros((k,))
	for i in range(k):
		cents[i] = minVal + gap * i


	return cents

source = """
	#include<stdio.h>
	__global__ void find_min_dist(float *dataSet, float *centroids, int *clusterState, int *numChange,int *k){
		int ind = blockIdx.x;
		float data = dataSet[ind];
		float minDist = 9999999;
		int findInd = 0;	

		for(int i = 0; i < *k; i++){
			float curDist = (data - centroids[i]) * (data - centroids[i]);
			if (curDist < minDist){
				minDist = curDist;
				findInd = i;
			}
			
		}
		if(int(clusterState[ind]) != findInd){
			clusterState[ind] = findInd;
			atomicAdd(numChange, 1);
		}
	}

	"""


def kmeans(dataSet,k):
	numSamples = dataSet.shape[0]
	clusterChanged = True
	clusterState = np.zeros((numSamples,),dtype='int32')
	centroids = initCents(dataSet,k)

	numIter = 0
	print "numSamples: " + str(numSamples)
	
	k_ptr = np.array([k],dtype="int32")

	# cuda only accept float32
	dataSet = dataSet.astype(np.float32)
	dataSet_gpu = cuda.mem_alloc(dataSet.nbytes)
	cuda.memcpy_htod(dataSet_gpu,dataSet)

	centroids = centroids.astype(np.float32)
	cents_gpu = cuda.mem_alloc(centroids.nbytes)
	cuda.memcpy_htod(cents_gpu,centroids)

	clusterState_gpu = cuda.mem_alloc(clusterState.nbytes)
	cuda.memcpy_htod(clusterState_gpu,clusterState)

	k_gpu = cuda.mem_alloc(k_ptr.nbytes)
	cuda.memcpy_htod(k_gpu,k_ptr)

	mod = SourceModule(source)
	find_min_dist = mod.get_function("find_min_dist")

	while clusterChanged:
		# numChange is 0 from the start of every iter
		numChange = np.array([0],dtype="int32")
		numChange_gpu = cuda.mem_alloc(numChange.nbytes)
		cuda.memcpy_htod(numChange_gpu, numChange)

		find_min_dist(dataSet_gpu,cents_gpu,clusterState_gpu,numChange_gpu,k_gpu, grid=(numSamples,1), block = (1,1,1))	
		pycuda.driver.Context.synchronize()
		cuda.memcpy_dtoh(numChange,numChange_gpu)

		print "iter " + str(numIter) + ", points chanegd " + str(numChange[0])
		cuda.memcpy_dtoh(clusterState, clusterState_gpu)
		
		#update centroids
		for j in range(k):
			pointsInCluster = dataSet[np.nonzero(clusterState == j)[0]]
			centroids[j] = np.mean(pointsInCluster)
		# transfer new centroids to gpu
		cuda.memcpy_htod(cents_gpu,centroids)
			
		numIter += 1
		if numChange[0] < numSamples * 0.01:
			break
	
	return centroids,clusterState












