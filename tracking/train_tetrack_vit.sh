# There are the detailed training settings for TETrack-ViT-B and TETrack-ViT-L.
# 1. download pretrained ViT-MAE models (mae_pretrain_vit_base.pth.pth/mae_pretrain_vit_large.pth) at https://github.com/facebookresearch/mae
# 2. set the proper pretrained ViT-MAE models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/tetrack_vit/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.


### Training TETrack-ViT-B-GOT
# python tracking/train.py --script tetrack_vit --config baseline_got --save_dir output --mode multiple --nproc_per_node 8


### Training TETrack-ViT-B
# python tracking/train.py --script tetrack_vit --config baseline --save_dir output --mode multiple --nproc_per_node 8


### Training TETrack-ViT-L-GOT
# python tracking/train.py --script tetrack_vit --config baseline_large_got --save_dir output --mode multiple --nproc_per_node 8


### Training TETrack-ViT-L
python tracking/train.py --script tetrack_vit --config baseline_large --save_dir output --mode multiple --nproc_per_node 8
