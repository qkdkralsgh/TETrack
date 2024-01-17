# Different test settings for TETrack-ViT-B, TETrack-ViT-L on LaSOT/TrackingNet/GOT10K


##########-------------- TETrack-B -----------------##########
# python tracking/test.py tetrack_vit baseline_got --dataset got_10k_test --threads 24 --num_gpus 8 --params__model TETrack_Base_got/TETrack_ep0490.pth.tar  # --viz_attn 1
# python lib/test/utils/transform_got10k.py --tracker_name tetrack_vit --cfg_name baseline_got


##########-------------- TETrack-L -----------------##########

### GOT10k test and pack
# python tracking/test.py tetrack_vit baseline_large_got --dataset got_10k_test --threads 24 --num_gpus 8 --params__model TETrack_Large_got/TETrack_ep0145.pth.tar --viz_attn 1
# python lib/test/utils/transform_got10k.py --tracker_name tetrack_vit --cfg_name baseline_large_got

## LaSOT test and evaluation
python tracking/test.py tetrack_vit baseline_large --dataset lasot --threads 24 --num_gpus 8 --params__model TETrack_Large_full/TETrack_ep0490.pth.tar --viz_attn 1
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large


### TrackingNet test and pack
# python tracking/test.py tetrack_vit baseline_large --dataset trackingnet --threads 24 --num_gpus 8 --params__model TETrack_Large_full/TETrack_ep0490.pth.tar --params__search_area_scale 4.6 # --viz_attn 1
# python lib/test/utils/transform_trackingnet.py --tracker_name tetrack_vit --cfg_name baseline_large


