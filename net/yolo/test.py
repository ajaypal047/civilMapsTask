from utils.im_transform import imcv2_recolor, imcv2_affine_trans
from utils.box import BoundBox, box_iou, prob_compare
import numpy as np
import cv2
import os
import json

def _fix(obj, dims, scale, offs):
	for i in range(1, 5):
		dim = dims[(i + 1) % 2]
		off = offs[(i + 1) % 2]
		obj[i] = int(obj[i] * scale - off)
		obj[i] = max(min(obj[i], dim), 0)

def preprocess(self, im, allobj = None):
	"""
	Takes an image, return it as a numpy tensor that is readily
	to be fed into tfnet. If there is an accompanied annotation (allobj),
	meaning this preprocessing is serving the train process, then this
	image will be transformed with random noise to augment training data, 
	using scale, translation, flipping and recolor. The accompanied 
	parsed annotation (allobj) will also be modified accordingly.
	"""	
	
	if type(im) is not np.ndarray:
		im = cv2.imread(im)
	


	if allobj is not None: # in training mode
		result = imcv2_affine_trans(im)
		im, dims, trans_param = result
		scale, offs, flip = trans_param
		for obj in allobj:
			_fix(obj, dims, scale, offs)
			if not flip: continue
			obj_1_ =  obj[1]
			obj[1] = dims[0] - obj[3]
			obj[3] = dims[0] - obj_1_
		im = imcv2_recolor(im)

	h, w, c = self.meta['inp_size']
	imsz = cv2.resize(im, (h, w))
	imsz = imsz / 255.
	imsz = imsz[:,:,::-1]
	if allobj is None: return imsz
	return imsz#, np.array(im) # for unit testing
	
_thresh = dict({
	'person': .2,
	'pottedplant': .1,
	'chair': .12,
	'tvmonitor': .13
})

def postprocess(self, net_out, im, save = True, check = False):
	"""
	Takes net output, draw predictions, save to disk
	"""
	meta, FLAGS = self.meta, self.FLAGS
	threshold, sqrt = FLAGS.threshold, meta['sqrt'] + 1
	C, B, S = meta['classes'], meta['num'], meta['side']
	colors, labels = meta['colors'], meta['labels']
	boxesReturn = []

	boxes = []
	SS        =  S * S # number of grid cells
	prob_size = SS * C # class probabilities
	conf_size = SS * B # confidences for each grid cell
	#net_out = net_out[0]
	probs = net_out[0 : prob_size]
	confs = net_out[prob_size : (prob_size + conf_size)]
	cords = net_out[(prob_size + conf_size) : ]
	probs = probs.reshape([SS, C])
	confs = confs.reshape([SS, B])
	cords = cords.reshape([SS, B, 4])

	for grid in range(SS):
		for b in range(B):
			bx   = BoundBox(C)
			bx.c =  confs[grid, b]
			bx.x = (cords[grid, b, 0] + grid %  S) / S
			bx.y = (cords[grid, b, 1] + grid // S) / S
			bx.w =  cords[grid, b, 2] ** sqrt
			bx.h =  cords[grid, b, 3] ** sqrt
			p = probs[grid, :] * bx.c
			p *= (p > threshold)
			bx.probs = p
			boxes.append(bx)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)): boxes[i].class_num = c
		boxes = sorted(boxes, key = prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				if box_iou(boxi, boxj) >= .4:
						boxes[j].probs[c] = 0.

	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im

	h, w, _ = imgcv.shape
	resultsForJSON = []

	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = self.meta['labels'][max_indx]
		if max_prob > _thresh.get(label,threshold):
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1

			if(check==False):
				thick = int((h + w) // 150)
				mess = '{}'.format(label)
				if self.FLAGS.json:
					resultsForJSON.append({"label": mess, "confidence": float('%.2f' % max_prob), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
					continue
				cv2.rectangle(imgcv, 
					(left, top), (right, bot), 
					self.meta['colors'][max_indx], thick)
				
				cv2.putText(
					imgcv, mess, (left, top - 12), 
					0, 1e-3 * h, self.meta['colors'][max_indx],
					thick // 3)
			else:
				b.x = left
				b.y = top
				b.w = right
				b.h = bot
				boxesReturn.append(b)
	
	if(check==False):
		outfolder = os.path.join(FLAGS.test, 'out') 
		img_name = os.path.join(outfolder, im.split('/')[-1])
		if self.FLAGS.json:
			textJSON = json.dumps(resultsForJSON)
			textFile = os.path.splitext(img_name)[0] + ".json"
			with open(textFile, 'w') as f:
				f.write(textJSON)
			return	

		if not save: return imgcv
		
		cv2.imwrite(img_name, imgcv)
	else:
		return boxesReturn
