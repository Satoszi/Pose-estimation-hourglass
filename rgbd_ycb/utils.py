import random
import matplotlib.pyplot as plt
import numpy as np

def visualize_image_data(generator):
    fig = plt.figure(figsize=(16, 8))
    rows = 1
    columns = 4

    for i in range(4):
        img, mask = next(generator)
        fig.add_subplot(rows, columns, i+1)
        img_to_visualize = img/2+mask[:,:,0:3]
        img_to_visualize = np.clip(img_to_visualize, 0, 1)
        plt.imshow(img_to_visualize)

def visualize_image_depth_data(generator):
    fig = plt.figure(figsize=(16, 8))
    rows = 1
    columns = 4

    for i in range(4):
        img, mask = next(generator)
        fig.add_subplot(rows, columns, i+1)
        img_to_visualize = img[:,:,3:6]/2+mask[:,:,0:3]
        img_to_visualize = np.clip(img_to_visualize, 0, 1)
        plt.imshow(img_to_visualize)


class MetricsCalculator():
    def __init__(self,):
        pass
    
    def get_blob_center(self, heatmap):

        heatmap_size = heatmap.shape[0]
        heatmap = np.abs(heatmap)
        heatmap = heatmap*heatmap*heatmap
        
        dx = np.array(np.arange(0,heatmap_size))
        dx = np.expand_dims(dx,0)
        dx = np.repeat(dx,heatmap_size, axis=0)
        dy = np.rot90(dx, k=3)

        heatmap_dx = dx*heatmap
        heatmap_dy = dy*heatmap
        
        sumed_heatmap = np.sum(heatmap)
        x = np.sum(heatmap_dx)/sumed_heatmap
        y = np.sum(heatmap_dy)/sumed_heatmap
        return np.array([x, y])

    def isNaN(self, num):
        return num != num

    def calc_blob_distances(self, mask_gt, mask_pred):
        distances = []
        for i in range(mask_gt.shape[-1]):
            heatmap_gt, heatmap_pred = mask_gt[:,:,i],mask_pred[:,:,i]
            #print("GT")
            p_gt = self.get_blob_center(heatmap_gt)
            #print("PRED")
            p_pred = self.get_blob_center(heatmap_pred)
            distance = np.linalg.norm(p_gt-p_pred)
            if self.isNaN(distance):
                distance = None
            distances.append(distance)
        cleaned_distances = [x for x in distances if x != None]
        return cleaned_distances
            
    def calc_oks_metric(self, model_to_eval, val_generator, probes):
        
        x = []
        y_gt = []
        for probe in range(probes):
            x1, y_gt1 = next(val_generator)
            x.append(x1)
            y_gt.append(y_gt1)
        all_frames_distances = []
        y_pred = model_to_eval.predict(np.array(x),verbose = 0)[1]

        for probe in range(probes):
            per_frame_distances = self.calc_blob_distances(y_gt[probe],y_pred[probe])
            per_frame_distances_avg = np.average(per_frame_distances)
            #print(per_frame_distances_avg)
            all_frames_distances.append(per_frame_distances_avg)
        average_dist = np.average(all_frames_distances)
        return average_dist


    def calc_pdj_metric(self, model_to_eval, val_generator, probes):
        
        x = []
        y_gt = []
        for probe in range(probes):
            x1, y_gt1 = next(val_generator)
            x.append(x1)
            y_gt.append(y_gt1)
        all_frames_distances = []
        y_pred = model_to_eval.predict(np.array(x),verbose = 0)[1]
        IMG_SIZE = x[0].shape[0]
        BLOBS_NUMBER = 7
        for probe in range(probes):
            per_frame_distances = self.calc_blob_distances(y_gt[probe],y_pred[probe])
            per_frame_distances_correct = np.sum([distance < IMG_SIZE*0.05 for distance in per_frame_distances])/BLOBS_NUMBER
            #print(per_frame_distances)
            all_frames_distances.append(per_frame_distances_correct)
        average_dist = np.average(all_frames_distances)

        return average_dist