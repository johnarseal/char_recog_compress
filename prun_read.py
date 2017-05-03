import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 
from bitarray import bitarray
import struct



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


caffe.set_mode_gpu()
layer_name = 're_fc8_128img'
caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/fc6_2048_input128/'
model_def = caffe_root + 'vgg13_deploy.prototxt'
model_weights = caffe_root + 'vgg13_iter_32000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)


param_fn = layer_name + ".params"
read_bitarr = bitarray(endian='big')
#read from file
with open(param_fn, 'rb') as f:
	read_bitarr.fromfile(f)


#fead the parameters vector
wts_mat = np.zeros_like(net.params[layer_name][0].data,dtype='int8')
wts_vec = wts_mat.flatten()
totalcnt = wts_vec.shape[0]
param_ind = 0
bit_ind = 0
print totalcnt
print len(read_bitarr)
while(param_ind < totalcnt):
	bit = read_bitarr[bit_ind]
	if bit == 0:
		wts_vec[param_ind] = 0
		bit_ind += 1
	else:
		bit_ind += 1
		byte = read_bitarr[bit_ind:bit_ind+8]
		wts_vec[param_ind] = struct.unpack(">b",byte)[0]
		bit_ind += 8 
	param_ind += 1



wts_vec = wts_vec.astype('float32')
wts_vec /= 1000
wts_vec = wts_vec.reshape(wts_mat.shape)
net.params[layer_name][0].data[...] = wts_vec

#test output
predict("/F/ZZ/char_patches/name_label_val.txt",net)


f.close()


