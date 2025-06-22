from pytorch_lightning.loggers import CSVLogger

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from app.module import DeepHashingModel
from app.dataset import ImageTripletDataModule


def main():
    config = {
        "dataset_name": "hyunlord/query_image_anchor_positive_large",
        "cache_dir": "./.cache",
        "model_name": "google/siglip2-base-patch16-384",
        "save_dir": "./checkpoints",

        "hash_dim": 128,
        "margin": 0.5,
        "lambda_ortho": 0.05,
        "lambda_distill": 1.0,
        "freeze_backbone_epochs": 0,

        "p": 4,
        "k": 4,
        "image_size": 384,
        "learning_rate": 3e-5,
        "epochs": 50,
        "num_workers": 28,
        "seed": 42,
    }
    pl.seed_everything(config['seed'], workers=True)

    datamodule = ImageTripletDataModule(config)
    model = DeepHashingModel(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['save_dir'],
        filename='best-model-{epoch:02d}-{val/hash_match_acc:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val/hash_match_acc',
        mode='max'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/hash_match_acc',
        patience=50,
        verbose=True,
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    csv_logger = CSVLogger("logs/", name="deep_hashing")

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        precision=32,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=csv_logger,
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
