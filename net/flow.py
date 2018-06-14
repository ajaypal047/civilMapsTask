import os
import time
import numpy as np
import tensorflow as tf
import pickle
from copy import deepcopy
from numpy.random import permutation as perm
from utils.udacity_voc_csv import udacity_voc_csv
from utils.box import BoundBox, box_iou, prob_compare
from utils.box import prob_compare2, box_intersection
from collections import OrderedDict

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op, self.summary_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        self.writer.add_summary(fetched[2], i)

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % 10 #(self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)
        
        if not ckpt:
            boxe=list()
            w = 1920
            h = 1200
            print("====================================")
            boxes = val(self)
            p=0
            for box in boxes:
                for b in box[1][0]:
                    k = b.probs
                    #print("boxes = self.x ", b.x, "self.y ", b.y, "self.w ", b.w,  "self.h ", b.h , "Class names ", self.meta['labels'][np.argmax(k)], "prob =", k)
                    boxe+=[[boxes[p][0],[w,h,[[self.meta['labels'][np.argmax(k)], b.x, b.y, b.w, b.h]]]]]
                p=p+1

            eval_list(boxe,self)



def eval_list(boxes,self):
    actuals = udacity_voc_csv(self.FLAGS.valAnn,self.meta['labels'])
    names = list()
    for box in actuals:
        names.append(box[0])

    imgName = list(OrderedDict.fromkeys(names))

    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5
    min_box_scale = 8. / 448

    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0

    groundBox = BoundBox(20)
    prdiction = BoundBox(20)

    for names in imgName:
        for boxgt in actuals:
            if(names[1:] == boxgt[0][1:]):
                total = total+1
                best_iou = 0
                for box in boxes:
                    if(box[0]== boxgt[0][1:] and box[1][2][0][0]==boxgt[1][2][0][0]):
                        proposals = proposals+1
                        box_gt = boxgt[1][2][0][1:5]
                        boxp = box[1][2][0][1:5]
                        groundBox.x = (box_gt[2]+box_gt[0])/2
                        groundBox.y = (box_gt[1]+box_gt[3])/2
                        groundBox.w = (box_gt[2]-box_gt[0])
                        groundBox.h = (box_gt[3]-box_gt[1])
                        prdiction.x = (boxp[2]+boxp[0])/2
                        prdiction.y = (boxp[1]+boxp[3])/2
                        prdiction.w = (boxp[2]-boxp[0])
                        prdiction.h = (boxp[3]-boxp[1])
                        iou = box_iou(groundBox, prdiction)
                        best_iou = max(iou, best_iou)
                    
                    if best_iou > iou_thresh:
                        avg_iou += best_iou
                        correct = correct+1
        if(proposals==0):
            precision = 0
        else:
            precision = 1.0*correct/proposals

        recall = 1.0*correct/total
        if(correct==0):
            fscore = 0
            IOU = 0
        else:
            fscore = 2.0*precision*recall/(precision+recall)
            IOU = avg_iou/correct
        proposals = 0
        total = 0
        print("Image no:",names[1:],"IOU: %f, Recal: %f, Precision: %f, Fscore: %f" % (IOU, recall, precision, fscore))
        

    


def predict(self):
    inp_path = self.FLAGS.test
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))
    all_inp_ = all_inp_[12:]
    batch = 1  

    boxes = []

    for j in range(len(all_inp_) // batch):
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            self.framework.postprocess(prediction,
                    os.path.join(inp_path, all_inp[i]), False, False)

        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))


def val(self):
    inp_path=self.FLAGS.val
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inp_))

    boxes = []

    for j in range(len(all_inp_) // batch):
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
        boxes1 = []
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            boxes.append(self.framework.postprocess(prediction, os.path.join(inp_path, all_inp[i]),False, True))
            boxes1.append([all_inp[i], boxes])

        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        return boxes1