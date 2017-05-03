import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 
from bitarray import bitarray
import struct


param_folder="params/"

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
		if cnt % 200 == 0:
			print "processed " + str(cnt) + " img"

	# calculate the avg
	print float(correct) / line_tot


def read_param_file(bitarr,param_mat,comp_type="int8"):
	if comp_type == "int8":
		pack = "<b"
		offset = 8
	elif comp_type == "int16":
		pack = "<h"
		offset = 16

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
			byte = bitarr[bit_ind:bit_ind+offset]
			wts_vec[param_ind] = struct.unpack(pack,byte)[0]
			bit_ind += offset
		param_ind += 1

	wts_vec = wts_vec.astype('float32')
	wts_vec /= 1024
	wts_mat = wts_vec.reshape(param_mat.shape)
	return wts_mat


def read_fc_layer(layer_name,net):
	param_fn = layer_name + ".params"
	bitarr = bitarray(endian='little')
	#read from file
	with open(param_folder+param_fn, 'rb') as f:
		bitarr.fromfile(f)
	#read the parameters vector
	param_mat = np.zeros_like(net.params[layer_name][0].data,dtype='int8')
	wts_mat = read_param_file(bitarr,param_mat)
	net.params[layer_name][0].data[...] = wts_mat


	# for the bias
	bias_param_fn = "bias_" + layer_name + ".params"
	bias_bitarr = bitarray(endian='little')
	#read from file
	with open(param_folder+bias_param_fn, 'rb') as f:
		bias_bitarr.fromfile(f)
	#read the parameters vector
	bias_param_mat = np.zeros_like(net.params[layer_name][1].data,dtype='int8')
	bias_wts_mat = read_param_file(bias_bitarr,bias_param_mat)
	net.params[layer_name][1].data[...] = bias_wts_mat


def read_conv_layer(layer_name,net):
	param_fn = layer_name + ".params"
	bitarr = bitarray(endian='little')
	#read from file
	with open(param_folder+param_fn, 'rb') as f:
		bitarr.fromfile(f)
	#read the parameters vector
	param_mat = np.zeros_like(net.params[layer_name][0].data,dtype='int16')
	wts_mat = read_param_file(bitarr,param_mat,'int16')
	net.params[layer_name][0].data[...] = wts_mat

	# for the bias
	bias_param_fn = "bias_" + layer_name + ".params"
	bias_bitarr = bitarray(endian='little')
	#read from file
	with open(param_folder+bias_param_fn, 'rb') as f:
		bias_bitarr.fromfile(f)
	#read the parameters vector
	bias_param_mat = np.zeros_like(net.params[layer_name][1].data,dtype='int16')
	bias_wts_mat = read_param_file(bias_bitarr,bias_param_mat,'int16')
	net.params[layer_name][1].data[...] = bias_wts_mat


caffe.set_mode_gpu()
caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/fc6_2048_input128/'
model_def = caffe_root + 'vgg13_deploy.prototxt'
#model_weights = caffe_root + 'vgg13_iter_32000.caffemodel'
net = caffe.Net(model_def,caffe.TEST)


layer_name_arr = net.params.keys()
for layer_name in layer_name_arr:
	print "reading layer " + layer_name
	if len(layer_name) > 4 and layer_name[:4] == "conv":
		read_conv_layer(layer_name,net)
		pass
	else:
		read_fc_layer(layer_name,net)
	
print "testing accuracy"
#test output
predict("/F/ZZ/char_patches/name_label_val.txt",net)


