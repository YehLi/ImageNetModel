# Usage
Install [MMDetection](https://github.com/open-mmlab/mmdetection).

# Data preparation
Prepare COCO according to [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

# Wave-ViT
## Training
```
dist_train.sh configs/wavevit/retinanet_wavevit_s_fpn_1x_coco.py 8
dist_train.sh configs/wavevit/retinanet_wavevit_b_fpn_1x_coco.py 8
dist_train.sh configs/wavevit/mask_rcnn_wavevit_s_fpn_1x_coco.py 8
dist_train.sh configs/wavevit/mask_rcnn_wavevit_b_fpn_1x_coco.py 8
dist_train.sh configs/wavevit/atss_wavevit_s_fpn_3x_mstrain_fp16.py 8
dist_train.sh configs/wavevit/gfl_wavevit_s_fpn_3x_mstrain_fp16.py 8
dist_train.sh configs/wavevit/cascade_mask_rcnn_wavevit_s_fpn_3x_mstrain_fp16.py 8
dist_train.sh configs/wavevit/sparse_rcnn_wavevit_s_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py 8
```

## Results on MSCOCO
**Object Detection task**
| Method | Backbone  |  AP  | AP50   |  AP75 | APs  | APm  | APl  | config file | 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| RetinaNet 1x       | Wave-ViT-S | 45.8 | 67.0 | 49.4 | 29.2 | 50.0 | 60.8 | [config](configs/wavevit/retinanet_wavevit_s_fpn_1x_coco.py) |
| RetinaNet 1x       | Wave-ViT-B | 47.2 | 68.2 | 50.9 | 29.7 | 51.4 | 62.3 | [config](configs/wavevit/retinanet_wavevit_b_fpn_1x_coco.py) |
| GFL                | Wave-ViT-S | 50.9 | 70.2 | 55.4 | 32.9 | 54.8 | 65.4 | [config](configs/wavevit/gfl_wavevit_s_fpn_3x_mstrain_fp16.py) |
| Sparse R-CNN       | Wave-ViT-S | 50.7 | 70.4 | 55.5 | 34.0 | 53.8 | 66.5 | [config](configs/wavevit/sparse_rcnn_wavevit_s_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) |
| ATSS               | Wave-ViT-S | 50.7 | 69.8 | 55.5 | 34.2 | 54.4 | 64.9 | [config](configs/wavevit/atss_wavevit_s_fpn_3x_mstrain_fp16.py) |
| Cascade Mask R-CNN | Wave-ViT-S | 52.1 | 70.7 | 56.6 | 34.1 | 55.6 | 67.6 | [config](configs/wavevit/cascade_mask_rcnn_wavevit_s_fpn_3x_mstrain_fp16.py) |

**Instance Segmentation task**
| Method |  Backbone  |  AP(b)  | AP50(b)   |  AP75(b) | AP(m)  | AP50(m)  | AP75(m)  | config file |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| Mask R-CNN 1x |  Wave-ViT-S | 46.6 | 68.7 | 51.2 | 42.4 | 65.5 | 45.8 | [config](configs/wavevit/mask_rcnn_wavevit_s_fpn_1x_coco.py) |
| Mask R-CNN 1x |  Wave-ViT-B | 47.6 | 69.1 | 52.4 | 43.0 | 66.4 | 46.0 | [config](configs/wavevit/mask_rcnn_wavevit_b_fpn_1x_coco.py) |

Wave-ViT pre-trained models can be downloaded [here](https://pan.baidu.com/s/1pWlyXad9IQIOMQNuotqPiw)

Access code for Baidu is **nets**

# Dual-ViT
## Training
```
dist_train.sh configs/dualvit/retinanet_dualvit_s_fpn_1x_coco.py 8
dist_train.sh configs/dualvit/retinanet_dualvit_b_fpn_1x_coco.py 8
dist_train.sh configs/dualvit/mask_rcnn_dualvit_s_fpn_1x_coco.py 8
dist_train.sh configs/dualvit/mask_rcnn_dualvit_b_fpn_1x_coco.py 8
dist_train.sh configs/dualvit/atss_dualvit_s_fpn_3x_mstrain_fp16.py 8
dist_train.sh configs/dualvit/gfl_dualvit_s_fpn_3x_mstrain_fp16.py 8
dist_train.sh configs/dualvit/cascade_mask_rcnn_dualvit_s_fpn_3x_mstrain_fp16.py 8
dist_train.sh configs/dualvit/sparse_rcnn_dualvit_b_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py 8
```

## Results on MSCOCO
**Object Detection task**
| Method | Backbone  |  AP  | AP50   |  AP75 | APs  | APm  | APl  | config file | 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| RetinaNet 1x       | Dual-ViT-S | 46.2 | 67.4 | 49.9 | 30.6 | 49.9 | 60.9 | [config](configs/dualvit/retinanet_dualvit_s_fpn_1x_coco.py) |
| RetinaNet 1x       | Dual-ViT-B | 47.4 | 68.1 | 51.2 | 29.6 | 51.9 | 63.1 | [config](configs/dualvit/retinanet_dualvit_b_fpn_1x_coco.py) |
| GFL                | Dual-ViT-S | 51.3 | 70.1 | 55.7 | 33.4 | 55.0 | 66.7 | [config](configs/dualvit/gfl_dualvit_s_fpn_3x_mstrain_fp16.py) |
| Sparse R-CNN       | Dual-ViT-S | 51.4 | 70.9 | 56.2 | 34.9 | 54.2 | 67.5 | [config](configs/dualvit/sparse_rcnn_dualvit_b_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) |
| ATSS               | Dual-ViT-S | 51.0 | 69.9 | 55.9 | 33.6 | 54.6 | 66.2 | [config](configs/dualvit/atss_dualvit_s_fpn_3x_mstrain_fp16.py) |
| Cascade Mask R-CNN | Dual-ViT-S | 52.4 | 71.0 | 56.9 | 36.2 | 56.0 | 68.0 | [config](configs/dualvit/cascade_mask_rcnn_dualvit_s_fpn_3x_mstrain_fp16.py) |

**Instance Segmentation task**
| Method |  Backbone  |  AP(b)  | AP50(b)   |  AP75(b) | AP(m)  | AP50(m)  | AP75(m)  | config file |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| Mask R-CNN 1x |  Dual-ViT-S | 46.5 | 68.3 | 51.2 | 42.2 | 65.3 | 46.1 | [config](configs/dualvit/mask_rcnn_dualvit_s_fpn_1x_coco.py) |
| Mask R-CNN 1x |  Dual-ViT-B | 48.4 | 69.9 | 53.3 | 43.4 | 66.7 | 46.8 | [config](configs/dualvit/mask_rcnn_dualvit_b_fpn_1x_coco.py) |

Dual-ViT pre-trained models can be downloaded [here](https://pan.baidu.com/s/14eufLJprx3Ug1cOPVc2H_g)

Access code for Baidu is **nets**
