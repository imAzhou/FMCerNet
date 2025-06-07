import os
import time
from torch.optim.lr_scheduler import LambdaLR
from mmengine.logging import MMLogger
from .tools import get_parameter_number
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from typing import Dict, List, Union
from mmengine.evaluator import Evaluator
from mmengine.registry import EVALUATOR


def get_logger(record_save_dir, model, print_cfg: Config):
    # set record files
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    files_save_dir = f'{record_save_dir}/{save_dir_date}'
    os.makedirs(files_save_dir, exist_ok=True)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    os.makedirs(pth_save_dir, exist_ok=True)
    # save config file
    config_file = os.path.join(files_save_dir, 'config.py')
    print_cfg.pop('backbone_cfgdict', None)
    print_cfg.dump(config_file)
    # save log file
    logger = MMLogger.get_instance(print_cfg.logger_name, log_file=f'{files_save_dir}/result.log')
    parameter_cnt = get_parameter_number(model)
    logger.info(f'total params: {parameter_cnt}')
    logger.info(f'update params:')
    for name,parameters in model.named_parameters():
        if parameters.requires_grad:
            logger.info(name)
    return logger,files_save_dir

def get_train_strategy(model, cfg):
    optimizerWrapper = build_optim_wrapper(model, cfg.optim_wrapper)

    '''slow start & fast decay'''
    def lr_lambda(epoch):
        current_lr = optimizerWrapper.param_groups[0]["lr"]
        if cfg.min_lr >= current_lr:
            decay_factor = cfg.min_lr / cfg.lr
            return decay_factor
        if epoch < cfg.warmup_epoch:
            return (epoch + 1) / cfg.warmup_epoch  # warm up 阶段线性增加
        else:
            # [base_lr*(args.gamma ** (epoch-args.warmup_epoch + 1))]
            # [0.005 *(0.9**i) for i in range(1,31)]
            return cfg.gamma ** (epoch-cfg.warmup_epoch + 1)
    lr_scheduler = LambdaLR(optimizerWrapper.optimizer, lr_lambda)

    return optimizerWrapper,lr_scheduler



def build_evaluator(evaluator: Union[Dict, List, Evaluator]) -> Evaluator:
    """Build evaluator.

    Examples of ``evaluator``::

        # evaluator could be a built Evaluator instance
        evaluator = Evaluator(metrics=[ToyMetric()])

        # evaluator can also be a list of dict
        evaluator = [
            dict(type='ToyMetric1'),
            dict(type='ToyEvaluator2')
        ]

        # evaluator can also be a list of built metric
        evaluator = [ToyMetric1(), ToyMetric2()]

        # evaluator can also be a dict with key metrics
        evaluator = dict(metrics=ToyMetric())
        # metric is a list
        evaluator = dict(metrics=[ToyMetric()])

    Args:
        evaluator (Evaluator or dict or list): An Evaluator object or a
            config dict or list of config dict used to build an Evaluator.

    Returns:
        Evaluator: Evaluator build from ``evaluator``.
    """
    if isinstance(evaluator, Evaluator):
        return evaluator
    elif isinstance(evaluator, dict):
        # if `metrics` in dict keys, it means to build customized evalutor
        if 'metrics' in evaluator:
            evaluator.setdefault('type', 'Evaluator')
            return EVALUATOR.build(evaluator)
        # otherwise, default evalutor will be built
        else:
            return Evaluator(evaluator)  # type: ignore
    elif isinstance(evaluator, list):
        # use the default `Evaluator`
        return Evaluator(evaluator)  # type: ignore
    else:
        raise TypeError(
            'evaluator should be one of dict, list of dict, and Evaluator'
            f', but got {evaluator}')
