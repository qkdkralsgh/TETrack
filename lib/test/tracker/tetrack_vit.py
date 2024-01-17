from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.tetrack.tetrack_vit import build_tetrack_vit
# from lib.models.tetrack.tetrack_vit_no_base import build_tetrack_vit
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from pathlib import Path
from lib.test.analysis.engine import get_attention_map


class TETrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TETrack, self).__init__(params)
        network = build_tetrack_vit(params.cfg, train=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        self.target_threshold = 63.0  # LaSOT large full 490 : 63.0 / Trackingnet large full 490 : 85.0 / Large_got 65.0  # got10k Large 70.0
        self.t_size = int(self.cfg.DATA.TEMPLATE.SIZE / 16)
        self.keep_ratio = self.cfg.MODEL.KEEP_RATIO 
        # for debug
        if params.debug:
            self.debug = True
        else:
            self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                
        if params.viz_attn:
            self.viz_attn = True
        else:
            self.viz_attn = False
        if self.viz_attn:
            self.save_dir = "viz_attn"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        self.template = template

        self.online_template = template
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        folder_name = info['folder_name']
        H, W, _ = image.shape
        self.frame_id += 1
        # print("frame id: {}".format(self.frame_id))
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            out_dict, _, attn_map, update_decision = self.network(self.template, self.online_template, search)
        
        ### Temporal filter ###
        self.pre_attn = attn_map
        
        # token_scores = out_dict['token_scores']
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        t_size = self.cfg.DATA.TEMPLATE.SIZE / 16

        if update_decision <= self.target_threshold:
            print(f'{folder_name} frame {self.frame_id} Update decision : {update_decision}')
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                       output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_template = self.preprocessor.process(z_patch_arr)

        # for debug  ### Edit by Minho 2023.07.13
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "tracking_result", "%s" % folder_name)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_file = os.path.join(save_path, "%04d.jpg" %  self.frame_id)
            cv2.imwrite(save_file, image_BGR)
        if self.viz_attn:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "tracking_result", "%s" % folder_name)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_file = os.path.join(save_path, "%04d.jpg" %  self.frame_id)
            cv2.imwrite(save_file, image_BGR)
            
            search_BGR = cv2.cvtColor(x_patch_arr, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(self.save_dir, "attn", "%s" % folder_name)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_attn_file = os.path.join(save_path, "%04d.jpg" %  self.frame_id)
            attention_image = get_attention_map(search_BGR, [attn_map[-1]])#,get_mask=True)
            cv2.imwrite(save_attn_file, attention_image)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def target_update_decision(self, token_scores):
        cnt = 0
        for i, scores in enumerate(token_scores):
            pred_ratio = scores.mean(1)
            if pred_ratio >= self.keep_ratio[i]:
                cnt += 1
            print(pred_ratio)
        return cnt


def get_tracker_class():
    return TETrack

