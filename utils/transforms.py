
import torchvision.transforms as T

# PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]
def build_transforms_base(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform



