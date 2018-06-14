"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import csv
import cv2
import numpy as np

def udacity_voc_csv(ANN):

    def pp(l): # pretty printing 
        for i in l: print('{}: {}'.format(i,l[i]))
    
    def parse(line): # exclude the xml tag
        x = line.decode().split('>')[1].decode().split('<')[0]
        try: r = int(x)
        except: r = x
        return r

    def _int(literal): # for literals supposed to be int 
        return int(float(literal))
    
    dumps = list()

    csv_fname = os.path.join(ANN)
    data = []
    line = 0
    with open(csv_fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|', )
        for row in spamreader:
            w = 1920
            h = 1200
            line = line+1      
            labels = row[1:]
            all = list()
            for i in range(0, len(labels), 5):
                xmin = int(labels[i])
                ymin = int(labels[i + 1])
                xmax = int(labels[i + 2])
                ymax = int(labels[i + 3])
                b = (xmin,
                     xmax,
                     ymin,
                     ymax)
                bb = convert_bbox((w, h), b)
                data.append(bb[2:])

           
    return np.array(data)
	
	
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def area(x):
    if len(x.shape) == 1:
        return x[0] * x[1]
    else:
        return x[:, 0] * x[:, 1]


def kmeans_iou(k, centroids, points, iter_count=0, iteration_cutoff=25, feature_size=13):

    best_clusters = []
    best_avg_iou = 0
    best_avg_iou_iteration = 0

    npoi = points.shape[0]
    area_p = area(points)  # (npoi, 2) -> (npoi,)

    while True:
        cen2 = centroids.repeat(npoi, axis=0).reshape(k, npoi, 2)
        cdiff = points - cen2
        cidx = np.where(cdiff < 0)
        cen2[cidx] = points[cidx[1], cidx[2]]

        wh = cen2.prod(axis=2).T  # (k, npoi, 2) -> (npoi, k)
        dist = 1. - (wh / (area_p[:, np.newaxis] + area(centroids) - wh))  # -> (npoi, k)
        belongs_to_cluster = np.argmin(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters_niou = np.min(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters = [points[belongs_to_cluster == i] for i in range(k)]
        avg_iou = np.mean(1. - clusters_niou)
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_clusters = clusters
            best_avg_iou_iteration = iter_count

        print("\nIteration {}".format(iter_count))
        print("Average iou to closest centroid = {}".format(avg_iou))
        print("Sum of all distances (cost) = {}".format(np.sum(clusters_niou)))

        new_centroids = np.array([np.mean(c, axis=0) for c in clusters])
        isect = np.prod(np.min(np.asarray([centroids, new_centroids]), axis=0), axis=1)
        aa1 = np.prod(centroids, axis=1)
        aa2 = np.prod(new_centroids, axis=1)
        shifts = 1 - isect / (aa1 + aa2 - isect)

        # for i, s in enumerate(shifts):
        #     print("{}: Cluster size: {}, Centroid distance shift: {}".format(i, len(clusters[i]), s))

        if sum(shifts) == 0 or iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break

        centroids = new_centroids
        iter_count += 1

    # Get anchor boxes from best clusters
    anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])
    anchors = anchors[anchors[:, 0].argsort()]
    print("k-means clustering pascal anchor points (original coordinates) \
    \nFound at iteration {} with best average IoU: {} \
    \n{}".format(best_avg_iou_iteration, best_avg_iou, anchors*feature_size))

    return anchors

if __name__ == "__main__":

    # examples
    # k, pascal, coco
    # 1, 0.30933335617, 0.252004954777
    # 2, 0.45787906725, 0.365835079771
    # 3, 0.53198291772, 0.453180358467
    # 4, 0.57562962803, 0.500282182136
    # 5, 0.58694643198, 0.522010174068
    # 6, 0.61789602056, 0.549904351137
    # 7, 0.63443906479, 0.569485509501
    # 8, 0.65114747974, 0.585718648162
    # 9, 0.66393113546, 0.601564171461

    # k-means picking the first k points as centroids
    img_size = 416
    k = 5

    random_data = np.random.random((1000, 2))
    centroids = np.random.random((k, 2))
    random_anchors = kmeans_iou(k, centroids, random_data)
	
    pascal_data = udacity_voc_csv("C:/Users/Arpit/Downloads/udacity.csv")
    centroids = pascal_data[np.random.choice(np.arange(len(pascal_data)), k, replace=False)]
    # centroids = pascal_data[:k]
    pascal_anchors = kmeans_iou(k, centroids, pascal_data, feature_size=img_size / 32)

    print('done')