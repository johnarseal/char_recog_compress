import caffe
import kmeans_cuda as km
import numpy as np
from prun_int_read import read_param_file,predict
from bitarray import bitarray
import math

param_folder = "kmeans_params/"
codebook_folder = "kmeans_params/codebook/"
NUM_K = 16
NUM_BITS = int(math.ceil((math.log(NUM_K,2))))




def prun(param_wts, threshold):
	param_wts[abs(param_wts) < threshold] = 0
	return param_wts


def prun_km_save_layer(layer_name,net,k):
	
	prefix = ["","bias_"]
	for wts_bias in range(2):
		param_fn = prefix[wts_bias] + layer_name + ".params"
		codebook_fn = prefix[wts_bias] + layer_name + ".codebook"

		origin_shape = net.params[layer_name][wts_bias].data.shape
		# do the pruning
		pruned_wts = prun(net.params[layer_name][wts_bias].data,0.01)
		# turn to vector
		wts_vec = pruned_wts.flatten()
		wts_ind = np.nonzero(wts_vec != 0)[0]
		nonzero_wts = wts_vec[wts_ind]
		
		# number of points that are non-zero
		numPts = len(nonzero_wts)
		
		# do kmeans
		cents, clusterState = km.kmeans(nonzero_wts,k)
		
		# recover the net params	
		param_vec = np.zeros_like(net.params[layer_name][wts_bias].data,dtype='float32').flatten()

		# build wts_ind_D
		wts_ind_D = {}
		for i in range(numPts):
			ind = int(clusterState[i])
			wts_ind_D[wts_ind[i]] = ind

		# prepare the bitarr to save
		save_bitarr = bitarray(endian='little')
		num_param = len(param_vec)
		for i in range(num_param):
			if i in wts_ind_D:
				ind = np.int16(wts_ind_D[i])
				cur_byte = bitarray('1',endian='little')
				cur_byte.frombytes(ind.tobytes())
				cur_byte = cur_byte[0:NUM_BITS+1]
				save_bitarr.extend(cur_byte)
			else:
				save_bitarr.append(0)
	
		with open(param_folder + param_fn,'wb') as f:
			save_bitarr.tofile(f)

		# save the codebook
		with open(codebook_folder + codebook_fn,'wb') as f:
			for i in range(len(cents)):
				f.write(str(i) + "\t" + str(cents[i]) + "\n")


if __name__ == "__main__":
	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/freq_6763_fc6_2048_input128/'
	model_def = caffe_root + "vgg13_deploy.prototxt"
	model_weights = caffe_root + 'vgg13_iter_16000.caffemodel'
	net = caffe.Net(model_def,model_weights,caffe.TEST)

	layer_arr = net.params.keys()
	
	# for every k we need to reload the parameters
	net = caffe.Net(model_def,model_weights,caffe.TEST)
		
	# iterate over layers
	for layer_name in layer_arr:
		prun_km_save_layer(layer_name,net,NUM_K)
		


