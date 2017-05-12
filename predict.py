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
	print float(correct) / line_tot


if __name__ == "__main__":
	caffe.set_mode_gpu()
	caffe_root = '/F/ZZ/caffe_workplace/models/vgg13_char_recog/hand_6763_fc6_2048_input128/'
	model_def = caffe_root + 'vgg13_deploy.prototxt'
	model_weights = caffe_root + 'vgg13_iter_32000.caffemodel'
	net = caffe.Net(model_def,model_weights,caffe.TEST)


	namef = open("/F/ZZ/wyz_ext/name_label_val.txt")
	lines = namef.readlines()
	correct = 0
	cnt = 0
	line_tot = len(lines)

	accuf = open("accu_analysis/cross_domain_result.txt",'w')
	labelRes = {}
	for line in lines:
		path,label = line[:-1].split()
		label = int(label)
		img = cv2.imread(path)
		net.blobs["data"].data[...] = preprocess(img)
		out = net.forward()
		if label not in labelRes:
			labelRes[label] = [0,0]
		if out['re_fc8_128img'].argmax() == label:
			correct += 1
			labelRes[label][0] += 1
		else:
			labelRes[label][1] += 1

		cnt += 1
		if cnt % 200 == 0:
			print "processed " + str(cnt) + " img"

	for label in labelRes:
		accuf.write(str(label) + "\t" + str(labelRes[label][0]) + "\t" + str(labelRes[label][1]) + "\n")



	# calculate the avg
	print float(correct) / line_tot
