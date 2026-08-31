model_type = 'ae'

inference_param = dict(
    exp_name='markcleaner_visible_inference',
    model_path='checkpoints/markcleaner.pth',
    sample_dir='outputs',
    mask_dilation=0,  # try 1-3 for antialiased watermark boundaries
    save_debug=False,
)

data_param = dict(
    path='/path/to/visible_eval',
    # False: white=keep, black=remove. True: white=remove, black=keep.
    mask_is_removal_region=False,
    num_workers=4,
    bs=1,
)

model_param = dict(
    image_size=(256, 256), patch_size=(16, 16), hidden_dim=12,
    use_dino_cls=True, use_dino_pred_loss=True, gaussian_per_patch=324,
    encoder_type='resnet',
    encoder_args=dict(use_skip=True, norm_layer='gn', up_norm_layer='gn'),
    overlap=True, overlap_pad=1, condition_type='direct',
)
