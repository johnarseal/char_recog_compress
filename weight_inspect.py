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

caffe.set_mode_gpu()
caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/fc6_2048_input128/'
model_def = caffe_root + 'vgg13_deploy.prototxt'
model_weights = caffe_root + 'vgg13_iter_32000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)

param_wts = net.params['re_fc8_128img'][0].data

'''
row,col = param_wts.shape
for i in range(row):
	for j in range(col):
		if type(param_wts[i][j]) != np.float32:
			print type(param_wts[i][j])
'''

wts_min = param_wts.min()
wts_max = param_wts.max()

print "min: " + str(wts_min)
print "max: " + str(wts_max)

wts_vec = param_wts.flatten()

print sum(abs(wts_vec) < 0.005)


