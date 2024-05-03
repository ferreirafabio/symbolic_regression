import torch

from torch.optim import AdamW
from workbench.train.optim.get_lr_schedule import get_learning_rate_schedule
from workbench.train.optim.adam_cpr import AdamCPR, group_parameters_for_cpr_optimizer
from workbench.train.optim.group_parameters import group_parameters_for_optimizer


def configure_optimizer(cfg_optim, model, logger):

    if cfg_optim.optimizer == 'AdamCPR':


        parameters = group_parameters_for_cpr_optimizer(model)

        optimizer = AdamCPR(parameters, lr=cfg_optim.lr, betas=cfg_optim.betas, kappa_init_param=cfg_optim.cpr_config.kappa_init_param,
                            kappa_init_method=cfg_optim.cpr_config.kappa_init_method)


    elif cfg_optim.optimizer == 'AdamW':
        parameters = group_parameters_for_optimizer(model, cfg_optim,
                                                    normalization_regularization=cfg_optim.normalization_regularization,
                                                    bias_regularization=cfg_optim.bias_regularization ,)
        optimizer = AdamW(parameters, lr=cfg_optim.lr, betas=cfg_optim.betas, weight_decay=cfg_optim.weight_decay)

    for i, g in enumerate(optimizer.param_groups):
        ntensors = len(g['params'])
        nparams = sum(p.numel() for p in g['params'])
        hparams = {k: v for k, v in g.items() if k != 'params'}
        logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

    if 'scheduler' not in cfg_optim:
        return optimizer
    else:
        lr_lambda = get_learning_rate_schedule(cfg_optim.scheduler)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return optimizer, lr_scheduler