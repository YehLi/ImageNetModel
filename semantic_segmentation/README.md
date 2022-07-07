# Usage
Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

# Wave-ViT Results on ADE20K
| Method |  Backbone | mIoU | config file | 
| :------------: | :------------: | :------------: | :------------: | 
| UPerNet | Wave-ViT-S  | 49.6 | [config](configs/wavevit/upernet_wavevit_s_512x512_160k_ade20k.py)   |
| UPerNet | Wave-ViT-B  | 51.5 | [config](configs/wavevit/upernet_wavevit_b_512x512_160k_ade20k.py)   |

# Dual-ViT Results on ADE20K
| Method |  Backbone | mIoU | config file | 
| :------------: | :------------: | :------------: | :------------: | 
| UPerNet | Dual-ViT-S  | 49.2 | [config](configs/dualvit/upernet_dualvit_s_512x512_160k_ade20k.py)   |
| UPerNet | Dual-ViT-B  | 51.7 | [config](configs/dualvit/upernet_dualvit_b_512x512_160k_ade20k.py)   |