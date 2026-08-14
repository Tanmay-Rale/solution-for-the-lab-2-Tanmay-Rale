import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torchvision.transforms as T
from torch import nn
from elec5308 import load_gtsrb, get_pretrained_resnet, Trainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE)
IMAGE_SIZE_TL = 64
train_transform_tl = T.Compose([
    T.Resize((IMAGE_SIZE_TL, IMAGE_SIZE_TL)),
    T.RandomRotation(degrees=8),
    T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
eval_transform_tl = T.Compose([
    T.Resize((IMAGE_SIZE_TL, IMAGE_SIZE_TL)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
train_loader_tl, val_loader_tl, test_loader_tl = load_gtsrb(
    root='./data', image_size=IMAGE_SIZE_TL, batch_size=64, val_fraction=0.15,
    train_transform=train_transform_tl, eval_transform=eval_transform_tl,
    num_workers=2, seed=2026, use_imagenet_stats=True,
)
model = get_pretrained_resnet(num_classes=43, freeze_backbone=True)
model = model.to(DEVICE)
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
trainer = Trainer(model, opt, nn.CrossEntropyLoss(), device=DEVICE, experiment=None, checkpoint_dir='./checkpoints/resnet')
h = trainer.fit(train_loader_tl, val_loader_tl, epochs=5, save_best=True)
print('phase1 best val', trainer._best_val_acc)
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
trainer = Trainer(model, opt, nn.CrossEntropyLoss(), device=DEVICE, experiment=None, checkpoint_dir='./checkpoints/resnet')
_ = trainer.fit(train_loader_tl, val_loader_tl, epochs=8, save_best=True)
print('phase2 best val', trainer._best_val_acc)
loss, acc, preds, labels = trainer.evaluate(test_loader_tl)
print('test acc', acc)
