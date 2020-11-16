import torch
from vit_pytorch import ViT

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 256, 256, 256)
mask = torch.ones(1, 8, 8, 8).bool()  # optional mask, designating which patch to attend to

print(img.shape)
preds = v(img, mask=mask)
print(preds.shape)
