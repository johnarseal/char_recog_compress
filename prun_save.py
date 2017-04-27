import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 
from bitarray import bitarray


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

param_wts = net.params[layer_name][0].data


wts_vec = param_wts.flatten()
wts_vec[abs(wts_vec) < 0.01] = 0


# multiply by 1000 and then convert to int8
wts_vec *= 1000
wts_vec = wts_vec.astype('int8')

# starting to save the weights in bytes
# the maximum length of bytes we need
save_bitarr = bitarray(endian='big')
for wts in wts_vec:
	if wts == 0:
		save_bitarr.append(0)
	else:
		cur_byte = bitarray('1',endian='big')
		cur_byte.frombytes(wts.tobytes())
		save_bitarr.extend(cur_byte)

f = open(layer_name+'.params','wb')
save_bitarr.tofile(f)
f.close()


