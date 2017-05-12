import kmeans_cuda
import caffe
import kmeans as km
from prun_int_read import predict

if __name__ == "__main__":
	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/freq_6763_fc6_2048_input128/'
	model_def = caffe_root + "vgg13_deploy.prototxt"
	model_weights = caffe_root + 'vgg13_iter_16000.caffemodel'
	net = caffe.Net(model_def,model_weights,caffe.TEST)

	layer_name = "re_fc8_128img"

	
	wts = net.params[layer_name][0].data
	origin_shape = wts.shape
	wts = wts.flatten()
	cents, clusterState = kmeans_cuda.kmeans(wts,8)
	numPts = len(wts)
	for i in range(numPts):
		ind = int(clusterState[i])
		wts[ind] = cents[ind]

	net.params[layer_name][0].data[...] = wts.reshape(origin_shape)
	
	predict("/F/ZZ/freq_wyz_ext/name_label_val.txt",net)


	'''
	numPts = len(wts)
	wts_mat = wts.reshape((numPts,1))
	km.kmeans(wts_mat,8)
	'''
	print "finished"
