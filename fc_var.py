import caffe
import cv2
import numpy as np

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
caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/'
model_def = caffe_root + 'vgg13_deploy.prototxt'
model_weights = caffe_root + 'snapshot_fc6_2048_input128_ft16k/vgg13_iter_12000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)
layer_name = 'fc7_128img'

fc_sum = np.zeros_like(net.blobs[layer_name].data[0],np.float64)
namef = open("/F/ZZ/char_patches/name_label_train.txt")
lines = namef.readlines()
cnt = 0
needed = 10000
for line in lines:
	if cnt == needed:
		break

	path = line.split()[0]
	img = cv2.imread(path)
	net.blobs["data"].data[...] = preprocess(img)
	net.forward()
	fc_sum += net.blobs[layer_name].data[0]		

	cnt += 1
	if cnt % 200 == 0:
		print "processed " + str(cnt) + " img"

# calculate the avg
fc_avg = fc_sum / needed


cnt = 0
fc_var =  np.zeros_like(net.blobs[layer_name].data[0],np.float64)
for line in lines:
	if cnt == needed:
		break
	path = line.split()[0]
	img = cv2.imread(path)
	net.blobs["data"].data[...] = preprocess(img)
	net.forward()
	diff = net.blobs[layer_name].data[0] - fc_avg
	sq = diff * diff
	fc_var += sq
	cnt += 1
	if cnt % 200 == 0:
		print "processed " + str(cnt) + " img"

fc_var /= (needed - 1)
fc_var.sort()


valid = 0
for n in fc_var:
	if n < 0.1:
		valid += 1
print valid





