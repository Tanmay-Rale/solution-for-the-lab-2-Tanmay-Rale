import json
from pathlib import Path

nb_path = Path(r'c:\Users\owner\Desktop\IIP lab2\ELEC5308-Lab2-tanmay-rale\lab2_notebook.ipynb')
with nb_path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = '''# Train the optional transfer-learning model
if RUN_TRANSFER_LEARNING:
    criterion_tl = nn.CrossEntropyLoss()

    # Phase 1: train the new classification head while the pretrained backbone stays frozen.
    optimiser_tl = torch.optim.Adam(
        filter(lambda p: p.requires_grad, resnet_model.parameters()),
        lr=LR_TL,
        weight_decay=1e-4,
    )

    trainer_tl = Trainer(
        model=resnet_model,
        optimiser=optimiser_tl,
        criterion=criterion_tl,
        device=DEVICE,
        experiment=experiment_tl,
        checkpoint_dir="./checkpoints/resnet",
    )

    history_tl = trainer_tl.fit(
        train_loader=train_loader_tl,
        val_loader=val_loader_tl,
        epochs=5,
        save_best=True,
    )

    # Phase 2: unfreeze the whole ResNet and fine-tune at a much smaller learning rate.
    for p in resnet_model.parameters():
        p.requires_grad = True

    optimiser_tl = torch.optim.Adam(
        resnet_model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    trainer_tl = Trainer(
        model=resnet_model,
        optimiser=optimiser_tl,
        criterion=criterion_tl,
        device=DEVICE,
        experiment=experiment_tl,
        checkpoint_dir="./checkpoints/resnet",
    )

    history_tl = trainer_tl.fit(
        train_loader=train_loader_tl,
        val_loader=val_loader_tl,
        epochs=10,
        save_best=True,
    )
    trainer_tl.load_best()
else:
    criterion_tl = None
    optimiser_tl = None
    trainer_tl = None
    history_tl = None
'''

for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'Train the optional transfer-learning model' in src:
        cell['source'] = new_cell.splitlines(keepends=True)
        break
else:
    raise RuntimeError('Could not find the transfer-learning training cell in the notebook.')

with nb_path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Updated transfer-learning notebook cell.')
