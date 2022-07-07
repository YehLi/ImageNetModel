# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='WaveViT',
        stem_hidden_dim=32, 
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14], 
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0.1, 
        depths=[3, 4, 6, 3],
        sr_ratios=[4, 2, 1, 1], 
        use_checkpoint=False),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 320, 448],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))