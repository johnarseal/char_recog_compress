import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 
from bitarray import bitarray
import struct
from prun_kmeans_cuda_save import NUM_BITS


param_folder="kmeans_params/"
codebook_folder = "kmeans_params/codebook/"

def normalize(img):
	nor_img = np.float32(img)
	nor_img = nor_img - 128
	nor_img /= 255.0
	return nor_img

def preprocess(img):
	img = cv2.resize(img,(128,128),0,0,cv2.INTER_LINEAR)
	img = normalize(img)
	img = np.transpose(img,(2,0,1))
	return img

def predict(fn,net):
	namef = open(fn)	
	lines = namef.readlines()

	correct = 0
	cnt = 0
	line_tot = len(lines)
	for line in lines:
		path,label = line[:-1].split()
		label = int(label)
		img = cv2.imread(path)
		net.blobs["data"].data[...] = preprocess(img)
		out = net.forward()
		if out['re_fc8_128img'].argmax() == label:
			correct += 1

		cnt += 1
		if cnt % 2000 == 0:
			print "processed " + str(cnt) + " img"

	# calculate the avg
	accu = float(correct) / line_tot
	print accu
	return accu


def read_km_param_file(bitarr,param_mat,codeBook):

	wts_vec = param_mat.flatten()
	totalcnt = wts_vec.shape[0]
	param_ind = 0
	bit_ind = 0
	while(param_ind < totalcnt):
		bit = bitarr[bit_ind]
		if bit == 0:
			wts_vec[param_ind] = 0
			bit_ind += 1
		else:
			bit_ind += 1
			offset = NUM_BITS
			num = bitarr[bit_ind:bit_ind+offset]
			# extend zero
			num.extend(('0' * (16 - offset)))
			ind = struct.unpack("<h",num)[0]
			wts_vec[param_ind] = codeBook[ind]
			bit_ind += offset

		param_ind += 1

	wts_mat = wts_vec.reshape(param_mat.shape)
	return wts_mat


def read_layer(layer_name,net):
	
	prefix = ["","bias_"]
	for bias_wts in range(2):
		cbDict = {}
		param_fn = prefix[bias_wts] + layer_name + ".params"
		codebook_fn = prefix[bias_wts] + layer_name + ".codebook"
		
		# read codebook and build the dict
		with open(codebook_folder + codebook_fn,'rb') as f:
			lines = f.readlines()
			for line in lines:
				ind,val = line[:-1].split()
				cbDict[int(ind)] = float(val)

		bitarr = bitarray(endian='little')
		
		# read from file
		with open(param_folder+param_fn,'rb') as f:
			bitarr.fromfile(f)
		param_mat = np.zeros_like(net.params[layer_name][bias_wts].data,dtype='float32')
		wts_mat = read_km_param_file(bitarr,param_mat,cbDict)
		net.params[layer_name][bias_wts].data[...] = wts_mat



if __name__ == "__main__":

	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/freq_6763_fc6_2048_input128/'
	model_def = caffe_root + 'vgg13_deploy.prototxt'
	model_weights = caffe_root + 'vgg13_iter_16000.caffemodel'
	net = caffe.Net(model_def,caffe.TEST)


	layer_name_arr = net.params.keys()
	for layer_name in layer_name_arr:
		print "reading layer " + layer_name
		read_layer(layer_name,net)	
	
	#test output
	predict("/F/ZZ/freq_wyz_ext/name_label_minival.txt",net)



