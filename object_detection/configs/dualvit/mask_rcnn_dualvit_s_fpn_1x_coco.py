_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained='Pretrained/dualvit_s.pth',
    backbone=dict(
        type='dualvit_s',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 448],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

fp16 = dict(loss_scale=512.)
find_unused_parameters = True
