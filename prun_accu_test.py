import caffe
import cv2
import numpy as np
import matplotlib as mpl  
import matplotlib.pyplot as plt 


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
caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/fc6_2048_input128/'
model_def = caffe_root + 'vgg13_deploy.prototxt'
model_weights = caffe_root + 'vgg13_iter_32000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)

param_wts = net.params['fc7_128img'][0].data

'''
row,col = param_wts.shape
for i in range(row):
	for j in range(col):
		if type(param_wts[i][j]) != np.float32:
			print type(param_wts[i][j])
'''

wts_vec = param_wts.flatten()
wts_vec[abs(wts_vec) < 0.01] = 0
wts_vec *= 1000
wts_vec = [int(x) for x in wts_vec]
wts_vec = np.array(wts_vec,dtype='float32')
wts_vec /= 1000
new_wts = wts_vec.reshape(4096,2048)

net.params['fc7_128img'][0].data[...] = new_wts
predict("/F/ZZ/char_patches/name_label_val.txt",net)






