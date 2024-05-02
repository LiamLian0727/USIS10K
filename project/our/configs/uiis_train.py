_base_ = ['anchor_net.py']

work_dir = './work_dirs/UIIS10K/adapter-baseline'

## ---------------------- MODEL ----------------------

crop_size = (512, 512)
num_classes = 10
# sam base model
sam_pretrain_name = "/root/code/pretrain/sam-vit-huge"
sam_pretrain_ckpt_path = "/root/code/pretrain/sam-vit-huge/pytorch_model.bin"

batch_augments = [dict(
    type='BatchFixedSizePad',
    size=crop_size,
    img_pad_value=0,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False
)]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)

model = dict(
    data_preprocessor=data_preprocessor,
    patch_embed_grad=False,
    decoder_freeze=False,
    shared_image_embedding=dict(extra_config=dict(image_size=crop_size[0])),
    backbone=dict(extra_config=dict(image_size=crop_size[0])),
    # backbone=dict(
    #     _delete_=True,
    #     type='MMPretrainSamVisionEncoder',
    #     img_size=crop_size[0],
    #     pretrain_name=sam_pretrain_name,
    #     init_cfg=dict(type='Pretrained', checkpoint=sam_pretrain_ckpt_path),
    #     peft_config=dict(
    #         peft_type="LORA", r=16, target_modules=["qkv"], lora_alpha=32, lora_dropout=0.05, bias="none",
    #     ),
    # ),
    # neck=dict(
    #     feature_aggregator=dict(
    #         _delete_=True,
    #         type='CNNAggregator',
    #         in_channels=256,
    #         hidden_channels=512,
    #         out_channels=256,
    #     ),
    # ),
    roi_head=dict(bbox_head=dict(num_classes=num_classes)),
    train_cfg=dict(rcnn=dict(mask_size=crop_size))
)

## ---------------------- Dataset ----------------------

# dataset settings
dataset_type = 'MultiClassUIIS10KInsSegDataset'
data_root = 'data/UIIS10K'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor')
    )
]

batch_size = 2
num_workers = 8
persistent_workers = True
indices = None

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        ann_file='annotations/multiclass_train.json',
        data_prefix=dict(img='img'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        ann_file='annotations/multiclass_test.json',
        data_prefix=dict(img='img'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)
val_dataloader = test_dataloader

test_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    ann_file=data_root + '/annotations/multiclass_test.json',
    format_only=False,
    backend_args=backend_args,
)
val_evaluator = test_evaluator

## ---------------------- Optim ----------------------

max_epochs = 24
base_lr = 0.0002

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

find_unused_parameters = True

# learning rate
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.001,
#         begin=1,
#         end=max_epochs,
#         T_max=max_epochs,
#         by_epoch=True
#     )
# ]
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15, 21],
        gamma=0.1
    )
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)
