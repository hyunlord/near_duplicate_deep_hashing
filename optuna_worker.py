import warnings
warnings.filterwarnings("ignore", message="No device id is provided via")
warnings.filterwarnings(
    "ignore",
    message=".* exceeds limit of .* pixels.*",
    category=UserWarning,
)

from tqdm import tqdm

import optuna

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, TQDMProgressBar

from app.module import DeepHashingModel
from app.dataset import ImageTripletDataModule


class SimpleTqdmCallback:
    def __init__(self, total_trials: int):
        self._pbar = tqdm(total=total_trials, desc="Optuna Trials")

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        self._pbar.update(1)


class CustomPruningCallback(Callback):
    def __init__(self, trial, monitor):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score.item(), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial):
    config = {
        "dataset_name": "hyunlord/query_image_anchor_positive_large_384",
        "cache_dir": "./.cache",
        "model_name": "google/siglip2-base-patch16-384",
        "save_dir": "./checkpoints_nhl",

        "hash_hidden_dim": trial.suggest_categorical("hash_hidden_dim", [256, 512, 768]),
        "hash_dim": 128,
        "margin": trial.suggest_float("margin", 0.2, 1.2),
        "lambda_ortho": trial.suggest_float("lambda_ortho", 0.0, 0.1),
        "lambda_lcs": trial.suggest_float("lambda_lcs", 0.0, 2.0),
        "lambda_codebook": 0.5,

        "freeze_backbone_epochs": 0,
        "batch_groups": 4,
        "images_per_group": 5,  # 빠른 실험을 위한 축소
        "image_size": 224,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "epochs": 5,  # 빠른 실험
        "num_workers": 4,
        "seed": 42,
        "bit_list": [8, 16, 32],
    }
    pl.seed_everything(config["seed"], workers=True)
    model = DeepHashingModel(config)
    datamodule = ImageTripletDataModule(config)

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=1,
        logger=False,
        callbacks=[CustomPruningCallback(trial, monitor="val/32_pos_hash_acc"),
                   TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=5
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val/32_pos_hash_acc"].item()


if __name__ == "__main__":
    study = optuna.load_study(
        study_name="deep_hash_opt",
        storage="sqlite:////hanmail/users/rexxa.som/shared/optuna.db"
    )
    total_trials = 1
    simple_pb = SimpleTqdmCallback(total_trials=1)
    study.optimize(objective,
                   n_trials=total_trials,
                   callbacks=[simple_pb])
