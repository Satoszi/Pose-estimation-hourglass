import random
import numpy as np
import cv2

import os
import tensorflow as tf
from scipy.spatial.transform import Rotation

class AugmentationManager():

    def __init__(self, aug_pipeline):
        self.aug_pipeline = aug_pipeline
        self.methods_mapping = {"channel_shift": self.channel_shift,
                                "rotation": self.rotation,
                                "flip": self.flip,
                                "random_rectangles": self.random_rectangles,
                                }       

    def augment(self, img, mask, depth):

        for operation in self.aug_pipeline:
            aug_type = operation["aug_type"]
            parameters = operation["parameters"]
            if aug_type == "channel_shift" and random.random() < 0.7:
                img = self.channel_shift(img)
            if aug_type == "rotation" and random.random() < 0.7:
                img, mask, depth = self.rotation(img, mask, parameters[0], depth)
            if aug_type == "flip": 
                img, mask, depth = self.flip(img, mask, depth)
                pass
            if aug_type == "random_rectangles" and random.random() < 0.5: 
                img = self.random_rectangles(img, 
                                            parameters[0], 
                                            parameters[1], 
                                            parameters[2])
        return img, mask, depth

    # TODO: wrapper
    # def wrapper(func, args): 
    #     func(*args)
        
    def channel_shift(self, img):
        value1 = random.random()*1.+0.6
        value2 = random.random()*0.5-0.25
        img = img*value1 + value2
        #img[:,:,:][img[:,:,:]>1]  = 1
        #img[:,:,:][img[:,:,:]<0]  = 0
        if random.random() < 0.3:
            img = np.flip(img,2)
        return img

    def rotation(self, img, mask, angle, depth):
        angle = int(random.uniform(-angle, angle))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
        if depth is not None:
            depth = cv2.warpAffine(depth, M, (w, h))
        return img, mask, depth

    def flip(self, img, mask, depth):
        if random.random() < 0.5:
            img = np.flip(img,1)
            mask = np.flip(mask,1)
            if depth is not None:
                depth = np.flip(depth,1)
        return img, mask, depth

    def random_rectangles(self, img, number, size_perc, whole = True):
        size = img.shape[0]

        if random.random() < 0.5:
            for i in range(number):
                
                x2 = int(random.random()*(size*size_perc))
                x1 = int(random.random()*(size-1-x2))
                y2 = int(random.random()*(size*size_perc))
                y1 = int(random.random()*(size-1-y2))

                if random.random() < 0.15 and size_perc > 0.4 and size_perc < 0.51:
                    x3 = int(random.random()*(size-1-x2))
                    y3 = int(random.random()*(size-1-y2))
                    img[y3:y3+y2,x3:x3+x2] = img[y1:y1+y2,x1:x1+x2]

                if random.random() < 0.3 and size_perc > 0.25 and size_perc < 0.4:
                    x3 = int(random.random()*(size-1-x2))
                    y3 = int(random.random()*(size-1-y2))
                    img[y3:y3+y2,x3:x3+x2] = img[y1:y1+y2,x1:x1+x2]

                if random.random() < 0.6 and size_perc < 0.25:
                    x3 = int(random.random()*(size-1-x2))
                    y3 = int(random.random()*(size-1-y2))
                    img[y3:y3+y2,x3:x3+x2] = img[y1:y1+y2,x1:x1+x2]

        if random.random() < 0.25:
            for i in range(number):
                
                x2 = int(random.random()*(size*size_perc))
                x1 = int(random.random()*(size-1-x2))
                y2 = int(random.random()*(size*size_perc))
                y1 = int(random.random()*(size-1-y2))

                if whole and random.random() < 0.5:
                    r = random.random()
                    g = random.random()
                    b = random.random()
                    if random.random() < 0.7:
                        img[y1:y2+y1,x1:x2+x1] = [r,g,b]
                    else:
                        fill = int(random.random()*3)*1-1
                        img = np.array(img) # fix for flip() breaks the type
                        img = cv2.circle(img,(x1,y1),int(size_perc * size/2),(r,g,b), fill)
                if not whole and random.random() < 0.5:
                    img[y1:y2+y1,x1:x2+x1] *= random.random()+0.5 + random.random()-0.3



        return img


class CustomDataCollector():

    def __init__(self,IMG_SIZE, sub_dirs_to_load,custom_ycb_path):
        self.IMG_SIZE = IMG_SIZE
        self.frames_poses_dict={}
        power_screwdriver_name = "power_screwdriver"
        poses_file_name = "035_power_drill.txt"
        poses_file_path = os.path.join(custom_ycb_path,poses_file_name)
        self.frames_path = os.path.join(custom_ycb_path,power_screwdriver_name)
        self.poses = np.loadtxt(poses_file_path)

        # all_types = ['0006', '0009', '0010', '0012', '0018', '0024', '0030', '0038',
        #             '0050','0056','0059','0077','0081','0083','0086','0088','0011', '0037', '0054']
        self.all_types = sub_dirs_to_load
        self.dir_types = os.listdir(self.frames_path)

    def collect(self,):
        idx = 0

        for dir_type in self.dir_types:
            dir_1 = os.path.join(self.frames_path,dir_type)
            if dir_type in self.all_types:
                print(dir_type)
                self.frames_poses_dict[dir_type] = {}
                self.frames_poses_dict[dir_type]['color_frames'] = []
                self.frames_poses_dict[dir_type]['depth_frames'] = []
                self.frames_poses_dict[dir_type]['frames_idx'] = []
                self.frames_poses_dict[dir_type]['poses'] = []
                for file_name in (os.listdir(dir_1)):
                    if "color" in file_name:
                        if idx % 10 == 0:
                            frame_path = os.path.join(dir_1,file_name)
                            frame = cv2.imread(frame_path)
                            frame = cv2.resize(frame,(self.IMG_SIZE,self.IMG_SIZE))/255.
                            self.frames_poses_dict[dir_type]['color_frames'].append(frame)
                            self.frames_poses_dict[dir_type]['poses'].append(self.poses[idx])
                            self.frames_poses_dict[dir_type]['frames_idx'].append(idx)
                        idx += 1
                    if "depth" in file_name:
                        if idx % 10 == 0:
                            frame_path = os.path.join(dir_1,file_name)
                            frame = cv2.imread(frame_path)
                            frame = cv2.resize(frame,(self.IMG_SIZE,self.IMG_SIZE))/255.
                            self.frames_poses_dict[dir_type]['depth_frames'].append(frame)
            else:
                for file_name in os.listdir(dir_1):
                    if "color" in file_name:
                        idx += 1





K1 = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
               [0.000000e+00, 1.067487e+03, 2.413109e+02],
               [0.000000e+00, 0.000000e+00, 1.000000e+00]])

K2 = np.array([[1077.836, 0, 323.7872],
               [0, 1078.189, 279.6921],
               [0,        0,        1]])

verts = [
        [-0.01,0,0],
        [0.08,0.05,0.0],
        [-0.1,0.045,0.0],
        [-0.03,-0.11,0.023],
        [-0.03,-0.11,-0.023],
        [0.055,-0.11,0.023],
        [0.055,-0.11,-0.023]
        ]
verts  = np.array(verts)



class CustomDataLoader():

    def __init__(self,IMG_SIZE, augmentator):
        self.IMG_SIZE = IMG_SIZE
        self.augmentator = augmentator

    def get_x_y(self, verts, pose, idx):
        t = pose[4:,None]
        quat = -np.array([*pose[1:4], pose[0]])
        R = np.array(Rotation.from_quat(quat).as_matrix())

        K_ = K1 if idx <= 23820 else K2

        verts_2d = np.matmul(K_, np.matmul(R, verts.T) + t).T
        verts_2d = verts_2d[:,:2] / verts_2d[:,2,None]
        return verts_2d

    def get_mask(self, points, original_frame_shape):
    
        SIZE_X = original_frame_shape[1]
        SIZE_Y = original_frame_shape[0]
        joint_masks = []
        mask = np.zeros((7,self.IMG_SIZE,self.IMG_SIZE))
        for i, point in enumerate(points):
            
            x = point[0]
            y = point[1]

            x = int(x/(SIZE_X/self.IMG_SIZE))
            y = int(y/(SIZE_Y/self.IMG_SIZE))
            color = (255,255,255)
            mask[i] = cv2.circle(mask[i],(x,y),7,color, -1)
            joint_masks.append(mask[i])
            
        joint_masks = np.swapaxes(joint_masks, 0,-1)
        joint_masks = np.swapaxes(joint_masks, 0,1)
            
        return joint_masks/255.

    def get_power_driver_trio(self, frames_poses_dict,dir_type,idx, aug):
        points = self.get_x_y(verts, frames_poses_dict[dir_type]['poses'][idx],frames_poses_dict[dir_type]['frames_idx'][idx])
        mask = self.get_mask(points, original_frame_shape = [480,640])
        color_frame = frames_poses_dict[dir_type]['color_frames'][idx].copy()
        try:
            depth_frame = frames_poses_dict[dir_type]['depth_frames'][idx].copy()
        except:
            depth_frame = frames_poses_dict[dir_type]['depth_frames'][idx-1].copy()
        if aug:
            color_frame, mask, depth_frame = self.augmentator.augment(color_frame, mask, depth_frame)
        return color_frame, mask, depth_frame



class DataGenerator(tf.keras.utils.Sequence):        
    def __init__(self, batch_size, gen, size):
        self.batch_size = batch_size
        self.on_epoch_end()
        self.gen = gen
        self.size = size
    def __len__(self):
        return int(self.size/self.batch_size)

    def __getitem__(self, index):
        #index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        #batch = [self.indices[k] for k in index]
        
        x_train = []
        y_train = []
        for i in range(self.batch_size):
            x, y = next(self.gen)
            x_train.append(x)
            y_train.append(y)
            
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        return x_train, y_train

    def on_epoch_end(self):
        pass


class DataGeneratorBuilder():
    def __init__(self, 
                data_loader, 
                frames_poses_dict,
                dir_types_go_gen,
                batch_size = 32 ,
                frames_to_skip = 0, 
                aug = False):
        
        self.frames_poses_dict = frames_poses_dict
        self.dir_types_go_gen = dir_types_go_gen
        self.dataset_size = self.get_dataset_size()
        self.generator = self.power_driver_generator(data_loader, frames_poses_dict, dir_types_go_gen, frames_to_skip, aug)
        self.batch_size = batch_size

    def get_dataset_size(self):
        dataset_size = 0
        for dir_type in self.dir_types_go_gen:
            dataset_size += len(self.frames_poses_dict[dir_type]['color_frames'])
        return dataset_size*3

    def power_driver_generator(self, data_loader, frames_poses_dict, dir_types_go_gen, frames_to_skip = 0, aug = False):
        while True:
            dir_type = random.choice(dir_types_go_gen)
            frames_number = len(frames_poses_dict[dir_type]['color_frames'])
            frame_number = int(random.random()*frames_number)
            color_frame, mask, depth = data_loader.get_power_driver_trio(frames_poses_dict, dir_type, frame_number, aug = aug)
            color_depth_frame = np.concatenate((color_frame, depth), axis = 2)
            yield color_depth_frame, mask

    def build_tf_generator(self):
        tf_generator = DataGenerator(batch_size=self.batch_size, gen=self.generator, size = self.dataset_size)
        return tf_generator


    
    





