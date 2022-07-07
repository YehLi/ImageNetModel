# Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

or

```
pip install mmdet==2.13.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

# Wave-ViT Results on MSCOCO
**Object Detection task**
| Method | Backbone  |  AP  | AP50   |  AP75 | APs  | APm  | APl  | config file | 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| RetinaNet 1x       | Wave-ViT-S | 45.8 | 67.0 | 49.4 | 29.2 | 50.0 | 60.8 | config |
| RetinaNet 1x       | Wave-ViT-B | 47.2 | 68.2 | 50.9 | 29.7 | 51.4 | 62.3 | config |
| RetinaNet 3x       | Wave-ViT-S | 47.3 | 68.5 | 50.9 | 32.8 | 51.3 | 61.8 | config |
| RetinaNet 3x       | Wave-ViT-B | 47.9 | 68.9 | 51.3 | 32.0 | 51.9 | 62.7 | config |
| GFL                | Wave-ViT-S | 50.9 | 70.2 | 55.4 | 32.9 | 54.8 | 65.4 | config |
| Sparse R-CNN       | Wave-ViT-S | 50.7 | 70.4 | 55.5 | 34.0 | 53.8 | 66.5 | config |
| ATSS               | Wave-ViT-S | 50.7 | 69.8 | 55.5 | 34.2 | 54.4 | 64.9 | config |
| Cascade Mask R-CNN | Wave-ViT-S | 52.1 | 70.7 | 56.6 | 34.1 | 55.6 | 67.6 | config |


**Instance Segmentation task**
| Method |  Backbone  |  AP(b)  | AP50(b)   |  AP75(b) | AP(m)  | AP50(m)  | AP75(m)  | config file |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| Mask R-CNN 1x |  Wave-ViT-S | 46.6 | 68.7 | 51.2 | 42.4 | 65.5 | 45.8 | config |
| Mask R-CNN 1x |  Wave-ViT-B | 47.6 | 69.1 | 52.4 | 43.0 | 66.4 | 46.0 | config |
| Mask R-CNN 3x |  Wave-ViT-S | 48.1 | 69.7 | 52.7 | 42.7 | 66.4 | 45.8 | config |
| Mask R-CNN 3x |  Wave-ViT-B | 49.2 | 70.4 | 54.0 | 43.2 | 67.1 | 46.3 | config |

# Dual-ViT Results on MSCOCO
**Object Detection task**
| Method | Backbone  |  AP  | AP50   |  AP75 | APs  | APm  | APl  | config file | 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| RetinaNet 1x       | Dual-ViT-S | 46.2 | 67.4 | 49.9 | 30.6 | 49.9 | 60.9 | config |
| RetinaNet 1x       | Dual-ViT-B | 47.4 | 68.1 | 51.2 | 29.6 | 51.9 | 63.1 | config |
| RetinaNet 3x       | Dual-ViT-S | 47.5 | 68.5 | 51.2 | 32.2 | 51.8 | 62.0 | config |
| GFL                | Dual-ViT-S | 51.3 | 70.1 | 55.7 | 33.4 | 55.0 | 66.7 | config |
| Sparse R-CNN       | Dual-ViT-S | 51.4 | 70.9 | 56.2 | 34.9 | 54.2 | 67.5 | config |
| ATSS               | Dual-ViT-S | 51.0 | 69.9 | 55.9 | 33.6 | 54.6 | 66.2 | config |
| Cascade Mask R-CNN | Dual-ViT-S | 52.4 | 71.0 | 56.9 | 36.2 | 56.0 | 68.0 | config |

**Instance Segmentation task**
| Method |  Backbone  |  AP(b)  | AP50(b)   |  AP75(b) | AP(m)  | AP50(m)  | AP75(m)  | config file |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| Mask R-CNN 1x |  Dual-ViT-S | 46.5 | 68.3 | 51.2 | 42.2 | 65.3 | 46.1 | config |
| Mask R-CNN 1x |  Dual-ViT-B | 48.4 | 69.9 | 53.3 | 43.4 | 66.7 | 46.8 | config |
| Mask R-CNN 3x |  Dual-ViT-S | 48.9 | 70.4 | 53.6 | 43.4 | 67.4 | 46.6 | config |