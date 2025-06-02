import pandas as pd
import pickle
import glob

from pathlib import Path

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.train import start_training
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.evaluate import start_evaluation 
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo import get_dataset  # Add this import


Vector = list[str]
def start_ensemble_prediction(cfg: Config, run_dir: Path, time_series_sub_dirs: Vector, output_dir: Path, epoch: int = None): 
    pred_results_dict = {}
    for sub_dir in time_series_sub_dirs:
        cfg.update_config({'time_series_data_sub_dir': sub_dir})
        tester = get_tester(cfg=cfg, run_dir=run_dir, period = 'test', init_model=True)
        pred_results = tester.evaluate(epoch=epoch, save_results=False)
        pred_results_dict[sub_dir] = pred_results

    file_name = output_dir / f"{Path(cfg.test_basin_file).stem}_prediction_results.p"
    file_name.parent.mkdir(parents=True, exist_ok=True)
    with file_name.open("wb") as fp:
        pickle.dump(pred_results_dict, fp)

    return None 


def inspect_data(cfg: Config):
    """Inspect the input data before training."""
    dataset = get_dataset(cfg=cfg, is_train=True, period='train')
    print(f"Dataset shape: {len(dataset)} samples")
    for i in range(min(5, len(dataset))):  # Display the shape of the first 5 samples
        sample = dataset[i]
        print(f"Sample {i} shapes: {[v.shape for v in sample.values() if hasattr(v, 'shape')]}")

def main():
    config_file = Path('./nh_forecast_example/config.yml') 
    cfg = Config(config_file)

    ##### Ensure train_dir is set #####
    if cfg.train_dir is None:
        cfg.train_dir = Path(cfg.run_dir) / "train_data"
        cfg.train_dir.mkdir(parents=True, exist_ok=True)

    ##### Inspect Data #####
    inspect_data(cfg)

    ##### Training ##### 
    start_training(cfg=cfg)

    run_dir = glob.glob("./nh_forecast_example/runs/development_run*")
    run_dir = Path(sorted(run_dir)[-1]) # Latest
    cfg = Config(Path(run_dir) / "config.yml")

    ##### Prediction #####
    cfg.device = 'cpu'
    members = ['0']
    output_dir = Path('.')
    start_ensemble_prediction(cfg, run_dir=run_dir, time_series_sub_dirs=members, output_dir=output_dir, epoch=30)


if __name__ == '__main__': 
    main()

