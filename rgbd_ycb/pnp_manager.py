from ast import While
import numpy as np
import os
import cv2

def get_blob_center(heatmap):
    
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


def get_all_blobs_centers(mask):
    
    blobs_centers = []
    for idx in range(mask.shape[-1]):
        heatmap = mask[:,:,idx]
        blob_center = get_blob_center(heatmap)
        blobs_centers.append(blob_center)
        
    return blobs_centers

class FramesGeneratorManager():
    def __init__(self, dir_path, video_number, target_resolution, skip_frames):
        self.dir_path = dir_path
        self.video_number = video_number
        self.width = target_resolution[0]
        self.height = target_resolution[1]
        self.skip_frames = skip_frames
        self.frames_generator = self.create_frames_generator()

    def create_frames_generator(self):
        video_path = os.path.join(self.dir_path, self.video_number)
        while True:
            frames_count = 0
            for file_name in os.listdir(video_path):
                if "color" in file_name:
                    frames_count += 1
                    if frames_count % self.skip_frames == 0:
                        frame_path = os.path.join(video_path, file_name)
                        frame = cv2.imread(frame_path)
                        frame = cv2.resize(frame,(self.width,self.height))
                        yield frame

    def get_next_frame(self):
        return next(self.frames_generator)

powerscrewdriver_points_3d = np.array([
                    (-0.01,0,0),            # Button
                    (0.08,0.05,0.0),        # Back power driver
                    (-0.1,0.045,0.0),       # Front power driver
                    (-0.03,-0.11,0.023),    # Battery  
                    (-0.03,-0.11,-0.023),   # Battery
                    (0.055,-0.11,0.023),    # Battery
                    (0.055,-0.11,-0.023)    # Battery
                ])

class SolverPNP():
    def __init__(self, frame_shape,):
        self.model_points_3d = powerscrewdriver_points_3d
        self.size = frame_shape
        focal_len = self.size[1]
        center = (self.size[1]/2, self.size[0]/2)
        self.camera_matrix = np.array([
                                    [focal_len, 0, center[0]],
                                    [0, focal_len, center[1]],
                                    [0,  0,  1]
                                ], dtype = "double")
        self.dist_coeffs = np.zeros((4,1))

    def solvePnP(self, points_2d):
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, points_2d,self.camera_matrix, self.dist_coeffs, flags=1)
        if success:
            return rotation_vector, translation_vector
        else:
            return None, None
    
    def projectPoints(self, points_2d, points_3d_to_project):
        rotation_vector, translation_vector = self.solvePnP(points_2d)
        (points_2d, jacobian) = cv2.projectPoints(np.array([points_3d_to_project]), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
        return points_2d
