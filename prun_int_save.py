import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 
from bitarray import bitarray


param_folder = "6763_params_var4/"


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

def prun_and_int(param_wts,prun_threshold = 0.01):
	wts_vec = param_wts.flatten()
	wts_vec[abs(wts_vec) < prun_threshold] = 0

	# multiply by 1024 and then convert to int8 or int16
	wts_vec *= 1024
	# convert it to int16 to prevent loss of accuracy
	wts_vec = wts_vec.astype('int16')

	bitnum = [0,0,0,0]
	# starting to save the weights in bytes
	# the maximum length of bytes we need
	save_bitarr = bitarray(endian='little')
	for wts in wts_vec:
		if wts == 0:
			save_bitarr.append(0)
		else:
			# for int 4
			if wts < (2 ** 3 - 1) and wts > -(2 ** 3):
				bit_prefix = '100'
				offset = 7
				bitnum[0] += 1
			#for int 8
			elif wts < (2 ** 7 - 1) and wts > -(2 ** 7):
				bit_prefix = '101'
				offset = 11
				bitnum[1] += 1
			# for int 12
			elif wts < ( 2 ** 11 - 1) and wts > -(2 ** 11):
				bit_prefix = '110'
				offset = 15
				bitnum[2] += 1
			# for int 16
			else:
				bit_prefix = '111'
				offset = 19
				bitnum[3] += 1

			cur_byte = bitarray(bit_prefix,endian='little')
			cur_byte.frombytes(wts.tobytes())
			cur_byte = cur_byte[0:offset]
			save_bitarr.extend(cur_byte)
	for i in range(4):
		print float(bitnum[i]) / sum(bitnum)
	return save_bitarr

	
def save_layer(layer_name,net,threshold=0.01):
	
	prefix = ["","bias_"]
	for bias_wts in range(2):
		param_wts = net.params[layer_name][bias_wts].data
		bitarr = prun_and_int(param_wts,threshold)
		param_fn = prefix[bias_wts] + layer_name + ".params"
		with open(param_folder + param_fn,'wb') as f:
			bitarr.tofile(f)



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

if __name__ == "__main__":

	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/freq_6763_fc6_2048_input128/'
	model_def = caffe_root + 'vgg13_deploy.prototxt'
	model_weights = caffe_root + 'vgg13_iter_16000.caffemodel'
	net = caffe.Net(model_def,model_weights,caffe.TEST)
	layer_name_arr = net.params.keys()


	for layer_name in layer_name_arr:
		print "saving layer " + layer_name
		if len(layer_name) > 4 and layer_name[:4] == 'conv':
			if int(layer_name[4]) < 5:
				save_layer(layer_name,net,0)
			else:
				save_layer(layer_name,net)
		else:
			save_layer(layer_name,net)
	

