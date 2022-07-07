# Usage
### Requirement:
* PyTorch 1.8.0+
* Python3.8
* CUDA 10.1+
* [timm](https://github.com/rwightman/pytorch-image-models)==0.4.5
* [tlt](https://github.com/zihangJiang/TokenLabeling)==0.1.0
* pyyaml
* apex-amp

## CoTNet Results on ImageNet
| name | resolution | #params | FLOPs | Top-1 Acc. | Top-5 Acc. | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| CoTNet-50 | 224 | 22.2M | 3.3 | 81.3 | 95.6 | [GoogleDrive](https://drive.google.com/file/d/1SR5ezIu7LN943zHaUh4mC0ehxBVMqtfv/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1czr00SglgD8dNVK8jT1yLg) |
| CoTNeXt-50 | 224 | 30.1M | 4.3 | 82.1 | 95.9 | [GoogleDrive](https://drive.google.com/file/d/1j6b5D3xcZ5L_bHiQV0WfqyOieqZLVOCv/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1CeV9IH_P5N9yuO-wOpdGNw) |
| SE-CoTNetD-50 | 224 | 23.1M | 4.1 | 81.6 | 95.8 | [GoogleDrive](https://drive.google.com/file/d/1D2b5fr3lxpBpiFcCYBKngmmSgfVHt_56/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1s5Xg7AqzWuwFJUzOJDoo4Q) |
| CoTNet-101 | 224 | 38.3M | 6.1 | 82.8 | 96.2 | [GoogleDrive](https://drive.google.com/file/d/11jExbPEg4Eq5PApisZyE5k-1CbRYnsQb/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1Olpta0AV7N4OoiC8PB4BnA) |
| CoTNeXt-101 | 224 | 53.4M | 8.2 | 83.2 | 96.4 | [GoogleDrive](https://drive.google.com/file/d/1des5wgkBDUscQAs8IYOmKCKKUA46QLfJ/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1FM0QRZJee7uY7iKaEiUA-w) |
| SE-CoTNetD-101 | 224 | 40.9M | 8.5 | 83.2 | 96.5 | [GoogleDrive](https://drive.google.com/file/d/1PWIltQYpYZiDrpfZORRQzGzQeXVd2b2f/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1WGFzuwio5lWJKiOOJTnjdg) |
| SE-CoTNetD-152 | 224 | 55.8M | 17.0 | 84.0 | 97.0 | [GoogleDrive](https://drive.google.com/file/d/1MkMx0a8an3ikt6LZwClIOyabBnMfR91v/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/14mNVsSf-6WI3mxLN2WinWw) |
| SE-CoTNetD-152 | 320 | 55.8M | 26.5 | 84.6 | 97.1 | [GoogleDrive](https://drive.google.com/file/d/1E43T2jS37gR07p_FVWnjJNkMWeYMXgX9/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1kO5of8IPgL4HOudLeykS6w) |
Access code for Baidu is **cotn**

# Wave-ViT Results on ImageNet
| name | resolution | #params | FLOPs | Top-1 Acc. | Top-5 Acc. | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Wave-ViT-S | 224 | 22.7M | 4.7 | 83.9  | 96.6 | [Baidu](https://pan.baidu.com/s/1JfNgmBE5ieAGsBermjpSzQ) |
| Wave-ViT-S | 384 | 22.7M | 15.1 | 85.0 | 97.2 | [Baidu](https://pan.baidu.com/s/1RLxSDwVG1wQs29s0x3_6QA) |
| Wave-ViT-B | 224 | 33.5M | 7.2 |  84.8 | 97.1 | [Baidu](https://pan.baidu.com/s/16DLjX7dKaZuULn7OrxVqyw) |
| Wave-ViT-B | 384 | 33.5M | 23.4 | 85.6 | 97.4 | [Baidu](https://pan.baidu.com/s/1KYINKOLO4Bf9kabDMjN-TA) |
| Wave-ViT-L | 224 | 57.5M | 14.8 | 85.5 | 97.3 | [Baidu](https://pan.baidu.com/s/1L-ZQ1eiv4sdefvdi7Z4-bw) |
| Wave-ViT-L | 384 | 57.5M | 47.5 | 86.3 | 97.7 | [Baidu](https://pan.baidu.com/s/1BdQBbdUeeAo8CojNifURGw) |
Access code for Baidu is **nets**

# Dual-ViT Results on ImageNet
| name | resolution | #params | FLOPs | Top-1 Acc. | Top-5 Acc. | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Dual-ViT-S | 224 | 25.1M | 5.4 | 84.1  | 96.8 | Baidu |
| Dual-ViT-S | 384 | 25.1M | 18.4 | 85.2 | 97.3 | Baidu |
| Dual-ViT-B | 224 | 42.6M | 9.3 | 85.2  | 97.2 | Baidu |
| Dual-ViT-B | 384 | 42.6M | 31.9 | 86.0 | 97.6 | Baidu |
| Dual-ViT-L | 224 | 73.0M | 18.0 | 85.7 | 97.4 | Baidu |
| Dual-ViT-L | 384 | 73.0M | 60.5 | 86.5 | 97.8 | Baidu |
Access code for Baidu is **nets**