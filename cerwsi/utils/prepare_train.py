import os
import time
from mmengine.logging import MMLogger
from .tools import get_parameter_number
from mmengine.config import Config
from mmengine.optim import OptimWrapper,OptimWrapperDict,_ParamScheduler
from typing import Dict, List, Union,Sequence,Optional
from mmengine.evaluator import Evaluator
from mmengine.registry import EVALUATOR,PARAM_SCHEDULERS
import copy


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

def build_param_scheduler(optim_wrapper, scheduler, max_epochs, epoch_length):
    if not isinstance(optim_wrapper, OptimWrapperDict):
        # Since `OptimWrapperDict` inherits from `OptimWrapper`,
        # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
        # whether `self.optim_wrapper` is an `OptimizerWrapper` or
        # `OptimWrapperDict` instance. Therefore, here we simply check
        # self.optim_wrapper is not an `OptimWrapperDict` instance and
        # then assert it is an OptimWrapper instance.
        assert isinstance(optim_wrapper, OptimWrapper), (
            '`build_optimizer` should be called before'
            '`build_param_scheduler` because the latter depends '
            'on the former')
        param_schedulers = _build_param_scheduler(max_epochs, epoch_length,
            scheduler, optim_wrapper)  # type: ignore
        return param_schedulers
    else:
        param_schedulers = dict()
        for name, optimizer in optim_wrapper.items():
            if isinstance(scheduler, dict) and 'type' not in scheduler:
                # scheduler is a dict and each item is a ParamScheduler
                # object or a config to build ParamScheduler objects
                param_schedulers[name] = _build_param_scheduler(max_epochs, epoch_length,
                    scheduler[name], optimizer)
            else:
                param_schedulers[name] = _build_param_scheduler(max_epochs, epoch_length,
                    scheduler, optimizer)

        return param_schedulers

def _build_param_scheduler(
            max_epochs: int,
            epoch_length: int,
            scheduler: Union[_ParamScheduler, Dict, List],
            optim_wrapper: OptimWrapper) -> List[_ParamScheduler]:
        """Build parameter schedulers for a single optimizer.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.
            optim_wrapper (OptimWrapper): An optimizer wrapper object is
                passed to construct ParamScheduler object.

        Returns:
            list[_ParamScheduler]: List of parameter schedulers build from
            ``scheduler``.

        Note:
            If the train loop is built, when building parameter schedulers,
            it supports setting the max epochs/iters as the default ``end``
            of schedulers, and supports converting epoch-based schedulers
            to iter-based according to the ``convert_to_iter_based`` key.
        """
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)

                # Set default end
                default_end = max_epochs
                _scheduler.setdefault('end', default_end)

                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optim_wrapper,
                            epoch_length=epoch_length)))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')
        return param_schedulers

def lr_scheduler_step(param_schedulers, type):
    if param_schedulers is None:
        return
    if type == 'iter':
        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()

        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(param_schedulers, dict):
            for param_schedulers in param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')
        
    elif type == 'epoch':
        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if scheduler.by_epoch:
                    scheduler.step()

        if isinstance(param_schedulers, list):
            step(param_schedulers)
        elif isinstance(param_schedulers, dict):
            for param_schedulers in param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {param_schedulers}')

def scale_lr(real_bs,
            optim_wrapper: OptimWrapper,
            auto_scale_lr: Optional[Dict] = None) -> None:
        """Automatically scaling learning rate in training according to the
        ratio of ``base_batch_size`` in ``autoscalelr_cfg`` and real batch
        size.

        It scales the learning rate linearly according to the
        `paper <https://arxiv.org/abs/1706.02677>`_.

        Note:
            ``scale_lr`` must be called after building optimizer wrappers
            and before building parameter schedulers.

        Args:
            optim_wrapper (OptimWrapper): An OptimWrapper object whose
                parameter groups' learning rate need to be scaled.
            auto_scale_lr (Dict, Optional): Config to scale the learning
                rate automatically. It includes ``base_batch_size`` and
                ``enable``. ``base_batch_size`` is the batch size that the
                optimizer lr is based on. ``enable`` is the switch to turn on
                and off the feature.
        """
        if (auto_scale_lr is None or not auto_scale_lr.get('enable', False)):
            return None

        assert 'base_batch_size' in auto_scale_lr, \
            'Lack of `base_batch_size` in `auto_scale_lr`.'
        
        base_bs = auto_scale_lr['base_batch_size']
        ratio = float(real_bs) / float(base_bs)

        def _is_built(schedulers):
            if isinstance(schedulers, dict):
                return False if 'type' in schedulers else any(
                    _is_built(s) for s in schedulers.values())
            if isinstance(schedulers, list):
                return any(_is_built(s) for s in schedulers)
            return isinstance(schedulers, _ParamScheduler)

        assert isinstance(optim_wrapper, OptimWrapper), \
            '`scale_lr should be called after building OptimWrapper'
        wrappers = list(optim_wrapper.values()) if isinstance(
            optim_wrapper, OptimWrapperDict) else [optim_wrapper]
        for wrapper in wrappers:
            for group in wrapper.optimizer.param_groups:
                group['lr'] = group['lr'] * ratio


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
