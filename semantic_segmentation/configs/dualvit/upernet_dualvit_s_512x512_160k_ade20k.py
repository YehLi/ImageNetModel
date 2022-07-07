_base_ = [
    '../_base_/models/upernet_dualvit.py', '../_base_/datasets/cade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        stem_hidden_dim=32, 
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14], 
        drop_path_rate=0.15, #0.2, 
        depths=[3, 4, 6, 3],
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 448],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=150
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
#data=dict(samples_per_gpu=1, workers_per_gpu=0)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
fp16 = dict()
