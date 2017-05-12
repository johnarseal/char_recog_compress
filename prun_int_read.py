import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 
from bitarray import bitarray
import struct


param_folder="6763_params_var4/"

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


def read_param_file(bitarr,param_mat):
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
			bit_beg = bit_ind + 1
			bit_end = bit_ind + 3
			# 100: int4
			if bitarr[bit_beg:bit_end] == bitarray('00'):
				offset = 4
			# 101: int8
			elif bitarr[bit_beg:bit_end] == bitarray('01'):
				offset = 8
			# 110: int12
			elif bitarr[bit_beg:bit_end] == bitarray('10'):
				offset = 12
			# 111: int16	
			elif bitarr[bit_beg:bit_end] == bitarray('11'):
				offset = 16

			bit_ind += 3
			num = bitarr[bit_ind:bit_ind+offset]
			# get the sign and extend it
			sign = num[offset-1:offset]
			num.extend((sign * (16 - offset)))
			wts_vec[param_ind] = struct.unpack("<h",num)[0]
			bit_ind += offset

		param_ind += 1

	wts_vec = wts_vec.astype('float32')
	wts_vec /= 1024
	wts_mat = wts_vec.reshape(param_mat.shape)
	return wts_mat


def read_layer(layer_name,net):
	
	prefix = ["","bias_"]
	for bias_wts in range(2):
		param_fn = prefix[bias_wts] + layer_name + ".params"
		bitarr = bitarray(endian='little')
		# read from file
		with open(param_folder+param_fn,'rb') as f:
			bitarr.fromfile(f)
		param_mat = np.zeros_like(net.params[layer_name][bias_wts].data,dtype='int16')
		wts_mat = read_param_file(bitarr,param_mat)
		net.params[layer_name][bias_wts].data[...] = wts_mat

	

if __name__ == "__main__":

	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/freq_6763_fc6_2048_input128/'
	model_def = caffe_root + 'vgg13_deploy.prototxt'
	model_weights = caffe_root + 'vgg13_iter_16000.caffemodel'
	net = caffe.Net(model_def,model_weights,caffe.TEST)


	layer_name_arr = net.params.keys()
	for layer_name in layer_name_arr:
		print "reading layer " + layer_name
		read_layer(layer_name,net)	
	
	#test output
	predict("/F/ZZ/freq_wyz_ext/name_label_minival.txt",net)







