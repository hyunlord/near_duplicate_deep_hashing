import warnings
warnings.filterwarnings("ignore", message="No device id is provided via")
warnings.filterwarnings(
    "ignore",
    message=".* exceeds limit of .* pixels.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*It is recommended to use.*",
    module="pytorch_lightning.trainer.connectors.logger_connector.result"
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

        "hash_hidden_dim": trial.suggest_categorical("hash_hidden_dim", [256, 512, 768, 1024]),
        "margin": trial.suggest_float("margin", 0.1, 1.5),
        "lambda_ortho": trial.suggest_float("lambda_ortho", 0.0, 0.2),
        "lambda_lcs": trial.suggest_float("lambda_lcs", 0.0, 2.0),
        "lambda_cons": trial.suggest_float("lambda_cons", 0.0, 0.5),
        "lambda_quant": trial.suggest_float("lambda_quant", 0.0, 0.2),

        "batch_groups": 4,
        "images_per_group": 10,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "epochs": 3,
        "num_workers": 28,
        "seed": 42,
        "bit_list": [8, 16, 32, 48, 64, 128]
    }
    pl.seed_everything(config["seed"], workers=True)
    model = DeepHashingModel(config)
    datamodule = ImageTripletDataModule(config)

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=1,
        precision='16-mixed',
        logger=False,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        log_every_n_steps=1
    )
    trainer.fit(model, datamodule=datamodule)

    metrics = trainer.callback_metrics
    for name, tensor in metrics.items():
        trial.set_user_attr(name, float(tensor))
    final_score = float(metrics["val/64_final_score"])
    return final_score


if __name__ == "__main__":
    study = optuna.load_study(
        study_name="final_score3_deep_hash_opt",
        storage="sqlite:////hanmail/users/rexxa.som/shared/optuna.db"
    )
    total_trials = 50
    simple_pb = SimpleTqdmCallback(total_trials=total_trials)
    study.optimize(objective,
                   n_trials=total_trials,
                   callbacks=[simple_pb])
