import yaml
import torch
import torchvision
import logging

from nas_model import gen_nas_model
from darts_model import gen_darts_model
from mobilenet_v1 import MobileNetV1
# from resnet import resnet


logger = logging.getLogger()


def build_model(args, model_name, pretrained=False, pretrained_ckpt=''):
    if model_name.lower() == 'nas_model':
        # model with architectures specific in yaml file
        model = gen_nas_model(yaml.safe_load(open(args.model_config, 'r')), drop_rate=0, 
                              drop_path_rate=0, auxiliary_head=False)

    elif model_name.lower() == 'darts_model':
        # DARTS evaluation models
        model = gen_darts_model(yaml.safe_load(open(args.model_config, 'r')), args.dataset, drop_rate=args.drop, 
                                drop_path_rate=args.drop_path_rate, auxiliary_head=False)

    elif model_name.lower() == 'mobilenet_v1':
        # mobilenet v1
        model = MobileNetV1(num_classes=args.num_classes)

    elif model_name.startswith('tv_'):
        # build model using torchvision
        import torchvision
        model = getattr(torchvision.models, model_name[3:])(pretrained=pretrained)

    elif model_name.startswith('timm_'):
        # build model using timm
        import timm
        model = timm.create_model(model_name[5:], pretrained=pretrained, drop_path_rate=args.drop_path_rate)

    elif model_name.startswith('cifar_'):
        from .cifar import model_dict
        model_name = model_name[6:]
        model = model_dict[model_name](num_classes=args.num_classes)
    else:
        raise RuntimeError(f'Model {model_name} not found.')

    if pretrained and pretrained_ckpt != '':
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = \
                model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) != 0:
            logger.info(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.info(f'Unexpected keys in source state dict: {unexpected_keys}')

    return model


