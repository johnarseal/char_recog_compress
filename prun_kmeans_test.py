import caffe
import kmeans as km
import numpy as np
from prun_int_read import read_param_file,predict
from bitarray import bitarray

param_folder = "6763_params/"

def prun(param_wts, threshold):
	param_wts[abs(param_wts) < threshold] = 0
	return param_wts


def prun_km_layer(layer_name,net,k):

	for wts_bias in range(2):
		origin_shape = net.params[layer_name][wts_bias].data.shape
		# do the pruning
		pruned_wts = prun(net.params[layer_name][wts_bias].data,0.01)
		# turn to vector
		wts_vec = pruned_wts.flatten()
		wts_ind = np.nonzero(wts_vec != 0)[0]
		nonzero_wts = wts_vec[wts_ind]
		# number of points that are non-zero
		numPts = len(nonzero_wts)
		nonzero_mat = nonzero_wts.reshape((numPts,1))
		# do kmeans
		cents, clusterState = km.kmeans(nonzero_mat,k)
		
		# recover the net params	
		param_vec = np.zeros_like(net.params[layer_name][wts_bias].data,dtype='float32').flatten()
		for i in range(numPts):
			ind = int(clusterState[i,0])
			param_vec[wts_ind[i]] = cents[ind][0]
	
			
		net.params[layer_name][wts_bias].data[...] = param_vec.reshape(origin_shape)		


if __name__ == "__main__":
	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/freq_6763_fc6_2048_input128/'
	model_def = caffe_root + "vgg13_deploy.prototxt"
	model_weights = caffe_root + 'vgg13_iter_16000.caffemodel'
	net = caffe.Net(model_def,model_weights,caffe.TEST)

	log_f = open("prun_kmeans.log",'w')

	layer_arr = net.params.keys()
	k_arr = (8,16,32,64,128,)

	for k in k_arr:
		# for every k we need to reload the parameters
		net = caffe.Net(model_def,model_weights,caffe.TEST)
		
		# iterate over layers
		for layer_name in layer_arr:
			print "processing layer " + layer_name
			prun_km_layer(layer_name,net,k)
		
		# predict
		print "predicting for " + str(k)
		accu = predict("/F/ZZ/freq_wyz_ext/name_label_val.txt",net)
		log_f.write(str(k) + "\t" + str(accu) + "\n")
		log_f.flush()

	log_f.close()
