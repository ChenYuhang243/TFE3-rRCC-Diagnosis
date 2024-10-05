import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
login("huggingface_access_token")
UNI_CKPT_PATH = "/path/to/uni/model/checkpoint"
CONCH_CKPT_PATH = "/path/to/conch/model/checkpoint"
CHIEF_CKPT_PATH = "/path/to/chief/model/checkpoint"
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        from models.conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == "gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif model_name == "virchow":
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    elif model_name =="chief":
        from .ctran import ctranspath
        model = ctranspath()
        model.head = torch.nn.Identity()
        td = torch.load(CHIEF_CKPT_PATH)
        model.load_state_dict(td['model'],strict=True)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    
    if model_name == "virchow":
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                             std=constants['std'],
                                             target_img_size = target_img_size)
    return model, img_transforms
