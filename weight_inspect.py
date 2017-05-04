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

layer_name_arr = net.params.keys()
frec = open("/F/ZZ/fyp_workplace/wts_min_max.txt",'w')
for layer_name in layer_name_arr:
	print "inspecting " + layer_name

	param_wts = net.params[layer_name][0].data
	wts_vec = param_wts.flatten()
	param_wts = net.params[layer_name][1].data
	wts_vec = list(wts_vec)
	bias_wts_vec = list(param_wts.flatten())
	wts_vec.extend(bias_wts_vec)
	wts_vec = np.array(wts_vec)

	wts_min = wts_vec.min()
	wts_max = wts_vec.max()

	below_num = sum(abs(wts_vec) < 0.01)
	below_prob = float(below_num) / len(wts_vec)

	frec.write(layer_name + " " + str(wts_min) + " " + str(wts_max) + " " + str(below_num) + " " + str(below_prob) + "\n")

	plt.hist(wts_vec,bins=20)
	plt.plot()
	plt.savefig("/F/ZZ/fyp_workplace/" + layer_name + "_weights.png")
