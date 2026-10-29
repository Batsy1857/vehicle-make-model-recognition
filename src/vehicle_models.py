def configure_transfer_stage(model, model_key, stage):
    """
    Configure which model parameters are trainable.

    stage:
      - "head_only"
      - "partial_fine_tune"
    """

    if model_key not in {"xception", "convnext_tiny"}:
        raise ValueError(f"Unknown model: {model_key}")

    if stage not in {"head_only", "partial_fine_tune"}:
        raise ValueError(f"Unknown stage: {stage}")

    # 1. Freeze the entire network
    for param in model.parameters():
        param.requires_grad_(False)

    # 2. Unfreeze the final classifier
    classifier = model.get_classifier()

    for param in classifier.parameters():
        param.requires_grad_(True)

    # 3. For Stage 2, unfreeze final backbone layers
    if stage == "partial_fine_tune":

        if model_key == "xception":
            final_modules = [
                "block12",
                "conv3",
                "bn3",
                "conv4",
                "bn4",
            ]

            for name in final_modules:
                module = getattr(model, name)

                for param in module.parameters():
                    param.requires_grad_(True)

        elif model_key == "convnext_tiny":
            assert len(model.stages) == 4

            for param in model.stages[-1].parameters():
                param.requires_grad_(True)

    # 4. Count parameters
    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }

import torch.nn as nn

def keep_batchnorm_in_eval(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()