import os
import warnings
import torch
from mmengine.config import Config

from cerwsi.nets.HMIL.trainer_manager import TrainingManager

# Setup PyTorch backend
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 
torch.autograd.set_detect_anomaly = True
warnings.filterwarnings('ignore')


def main():
    """Main function"""
    
    try:
        # Load configuration
        cfg = Config.fromfile('cerwsi/nets/HMIL/configs/config_file.py')
        
        # Setup GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[device] device set to: {device}')
        
        # Create training manager
        trainer = TrainingManager(cfg, device)
        
        # Parse training folds
        training_folds = [int(f) for f in cfg.folds.split(',')]
        cross_val_flag = len(training_folds) > 1
        
        if cross_val_flag:
            print(f"[logs] training folds: {training_folds}")
            print(f"[logs] conducting {len(training_folds)}-fold cross-validation")
            results = trainer.run_cross_validation(training_folds)
            print(f"Cross-validation results: {results}")
        else:
            print(f"[logs] training fold: {training_folds}")
            results = trainer.run_single_training(training_folds[0])
            print(f"Training results: {results}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main()