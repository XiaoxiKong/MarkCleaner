import sys
import os
from src.dinov2.models.vision_transformer import vit_with_map, load_base_state
import torch
import torchvision.transforms as transforms

def create_model():
    model = vit_with_map(return_cls_token=True)
    # dino.py is in src/, so its parent is the code directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # The src/ upper level is the project root directory
    pretrained_path = os.path.join(project_root, 'pretrained', 'dinov2_vitb14_pretrain.pth')
    load_base_state(model, path=pretrained_path)

    return model


@torch.no_grad
def get_feature(imgs, model=None):
    if model is None:
        model = create_model()

    resize_trans = transforms.Resize((224,224))
    imgs = resize_trans(imgs)

    assert (next(model.parameters()).device == imgs.device)

    fea = model(imgs)
    return fea.detach()


if __name__ == '__main__':
    model = create_model().to('cuda')
    imgs = torch.rand(8,3,256,256).cuda()
    fea = get_feature(imgs, model=model)
    print(fea.device)
