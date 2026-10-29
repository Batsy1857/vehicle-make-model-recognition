from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from timm.data import resolve_model_data_config


def build_vehicle_transforms(cfg, model):
    pipeline = cfg["data_pipeline"]
    train_cfg = pipeline["train"]
    eval_cfg = pipeline["evaluation"]

    assert pipeline["color_mode"] == "RGB"
    assert pipeline["interpolation"] == "bicubic"

    # Normalization comes from the pretrained checkpoint.
    model_data = resolve_model_data_config(model)
    mean = model_data["mean"]
    std = model_data["std"]

    size = tuple(pipeline["input_size"])

    train_transform = T.Compose([
        T.RandomResizedCrop(
            size,
            scale=tuple(
                train_cfg["random_resized_crop"]["scale"]
            ),
            ratio=tuple(
                train_cfg["random_resized_crop"]["ratio"]
            ),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(
            p=train_cfg[
                "random_horizontal_flip_probability"
            ]
        ),
        T.ColorJitter(
            **train_cfg["color_jitter"]
        ),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_transform = T.Compose([
        T.Resize(
            eval_cfg["resize_shorter_side"],
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.CenterCrop(tuple(eval_cfg["center_crop"])),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return train_transform, eval_transform