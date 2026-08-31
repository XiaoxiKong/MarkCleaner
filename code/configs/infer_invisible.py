model_type = 'ae'

inference_param = dict(
    exp_name='markcleaner_inference',
    model_path='checkpoints/markcleaner.pth',
    sample_dir='outputs',
)

# Required layout is documented in ../../docs/DATA_FORMAT.md.
data_param = dict(
    path='/path/to/invisible_eval',
    num_workers=4,
    bs=8,
)

model_param = dict(
    image_size=(256, 256),
    patch_size=(16, 16),
    hidden_dim=12,
    use_dino_cls=True,
    use_dino_pred_loss=True,
    gaussian_per_patch=324,
    encoder_type='resnet',
    encoder_args=dict(use_skip=True, norm_layer='gn', up_norm_layer='gn'),
    overlap=True,
    overlap_pad=1,
    condition_type='direct',
)
