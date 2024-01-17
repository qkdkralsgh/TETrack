from typing import Any
import cv2
import glob
import numpy as np
import sys

root = "/home/adminuser/Minho/MixFormer_SPViT_target/viz_attn/"
attn_dir = root + "attn_dir/"
result_dir = root + "result_dir/"

got10k = "got-10k/"
lasot = "lasot/"
trackingnet = "trackingnet/"

datasets = [got10k, lasot, trackingnet]

fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')

class VideoWriter:
    def __init__(self):
        self.w = None
        self.h = None
        self.root = "/home/adminuser/Minho/MixFormer_SPViT_target/viz_attn/"
        self.attn_dir = root + "attn_dir/"
        self.result_dir = root + "result_dir/"

        got10k = "got-10k/"
        lasot = "lasot/"
        trackingnet = "trackingnet/"

        self.datasets = [got10k] #, lasot, trackingnet]

        self.fps = 10.0
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video_writer = None
        
        self.sky = (255, 255, 0)
        self.red = (0, 0, 255)
        self.font = cv2.FONT_HERSHEY_COMPLEX
        
        
    def __call__(self):
        
        # self.video_writer = cv2.VideoWriter(root + "MixViT_original.mp4", self.fourcc, self.fps, (1568, 720))  ### (1280, 720)
        self.video_writer = cv2.VideoWriter(root + "MixViT_Online_target.mp4", self.fourcc, self.fps, (2208, 1080))  ### (1920, 1080)
            
        if not self.video_writer.isOpened():
            print("Video open failed!")
            sys.exit()
        
        # for dataset in self.datasets:
            # if dataset != 'got-10k/':
            #     self.fps = 20.0
                
        for cnt, (result_folder, attn_folder) in enumerate(zip(sorted(glob.glob(result_dir+"/*")), sorted(glob.glob(attn_dir+"/*")))):
            for i, (result_file, attn_file) in enumerate(zip(sorted(glob.glob(result_folder+"/*")), sorted(glob.glob(attn_folder+"/*")))):
                
                result_img = cv2.imread(result_file)
                attn_img = cv2.imread(attn_file)
                result_img_h, result_img_w, _ = result_img.shape
                attn_img_h, attn_img_w, _ = attn_img.shape
                
                # if num == 0:
                #     self.h = result_img_h
                    
                #     ### version 1 ###
                #     self.w = result_img_w + attn_img_w
                    
                    ### version 2 ###
                    # self.w = result_img_w
                    
                    # self.video_writer = cv2.VideoWriter(root + "USTAM_%s_%d.mp4" % (dataset[:-1], cnt+1), self.fourcc, self.fps, (self.w, self.h))
                    # if not self.video_writer.isOpened():
                    #     print("Video open failed!")
                    #     sys.exit()
                        
                # new_array = np.zeros((result_img_h, self.w, 3), dtype=np.uint8)
                # new_array = np.zeros((720, 1568, 3), dtype=np.uint8)  ### (1280, 720)
                new_array = np.zeros((1080, 2208, 3), dtype=np.uint8)  ### (1920, 1080)
                
                ### version 1 ###
                new_array[:result_img_h, :result_img_w, :] = result_img
                new_array[:attn_img_h, result_img_w:, :] = attn_img
                new_array = cv2.putText(new_array, str(i+1), (10, 50), self.font, 2, self.sky, 3)
                new_array = cv2.putText(new_array, "Attention map", (result_img_w+1, 30), self.font, 1, self.red, 2)
                
                ### version 2 ###
                # new_array[:result_img_h, :result_img_w, :] = result_img
                # new_array[:attn_img_h, result_img_w-attn_img_w:result_img_w, :] = attn_img
                
                self.video_writer.write(new_array)
        self.video_writer.release()
                
def main():
    Video_writer = VideoWriter()
    Video_writer()
    
if __name__ == "__main__":
    main()