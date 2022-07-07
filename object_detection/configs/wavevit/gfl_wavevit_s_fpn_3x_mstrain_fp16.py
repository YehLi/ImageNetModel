_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    pretrained='Pretrained/wavevit_s.pth',
    backbone=dict(
        type='wavevit_s',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 448],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[[
             dict(type='CResize',
                  img_scale=[(480, 1200), (512, 1200), (544, 1200), (576, 1200),
                             (608, 1200), (640, 1200), (672, 1200), (704, 1200),
                             (736, 1200), (768, 1200), (800, 1200)],
                  multiscale_mode='value',
                  keep_ratio=True)],
             [
                 dict(type='CResize',
                      img_scale=[(400, 1200), (500, 1200), (600, 1200)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='CResize',
                      img_scale=[(480, 1200), (512, 1200), (544, 1200),
                                 (576, 1200), (608, 1200), (640, 1200),
                                 (672, 1200), (704, 1200), (736, 1200),
                                 (768, 1200), (800, 1200)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])

# do not use apex fp16
runner = dict(type='EpochBasedRunner', max_epochs=36)
fp16 = dict(loss_scale=512.)

# use apex fp16
#runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
#fp16 = None
#optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=1,
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,
#    use_fp16=True,
#)

find_unused_parameters = True
