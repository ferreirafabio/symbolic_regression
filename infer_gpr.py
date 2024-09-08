import os
import sys
import socket
import argparse
from collections import defaultdict
import collections
import yaml, tqdm, pathlib
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import torch.cuda
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import accelerate

# from gpr.data.generators import PolynomialGenerator
from gpr.data.loaders import SymPySimpleDataModule
from gpr.data.equation_master import EquationMaster
from gpr.model.gpr_transformer import GPRTransformer
from gpr.utils.configuration import Config
from gpr.utils.folder_manager import get_experiment_folder
from gpr.model.module.ce_loss import FlashCrossEntropyLoss



def bold(msg):
    return f"\033[1m{msg}\033[0m"


def teacher_forcing_evaluation(model, step, set_name, data_loader, sympy_data, e_master, accelerator, device, logger, tb_logger=None):


    loss_func = FlashCrossEntropyLoss(ignore_index=sympy_data.ignore_index,
                                      reduction='mean')
    model.eval()

    logger.info(f"Start teacher forcing evaluation for {set_name}!")
    tefo_stats = defaultdict(lambda: torch.tensor(0, device=device, dtype=torch.float))

    with torch.no_grad():
        for val_batch in data_loader:
            val_batch = sympy_data.batch_to_device(val_batch, device)
            with accelerator.autocast():
                logits = model(val_batch['mantissa'],
                               val_batch['exponent'],
                               val_batch['in_equation'])
                val_loss = loss_func(logits.view(-1, logits.size(-1)), val_batch['trg_equation'].view(-1))

            tefo_stats['loss'] += val_loss

            sample_count = val_batch['trg_len'].size(0)
            token_count = torch.sum(val_batch['trg_len'], dtype=torch.float)

            tefo_stats['log_probs'] += val_loss * token_count

            predicted_tokens = logits.argmax(dim=-1)
            correct_tokens = predicted_tokens == val_batch['trg_equation']
            ignore_mask = val_batch['trg_equation'] != sympy_data.ignore_index
            batch_accuracy = torch.sum(correct_tokens & ignore_mask, dim=-1) / val_batch['trg_len']
            batch_solved = torch.sum(batch_accuracy == 1.0)
            tefo_stats['accuracy'] += torch.sum(batch_accuracy)
            tefo_stats['solved'] += torch.sum(batch_solved)

            tefo_stats['tokens'] += token_count
            tefo_stats['samples'] += sample_count
            tefo_stats['num_batches'] += 1

            # compute MSE between predicted and true equation of the current val batch
            true_strs = sympy_data.indices_to_string(val_batch['trg_equation'])
            pred_strs = sympy_data.indices_to_string(predicted_tokens)

            batch_sum_mse, batch_sum_valid_eq = e_master.compute_mse(pred_strs, true_strs)
            tefo_stats['mse'] += batch_sum_mse
            tefo_stats['valid'] += batch_sum_valid_eq
            # equation_mse += batch_sum_mse
            # valid_equation += batch_sum_valid_eq

    gathered_tefo_stats = {k: accelerator.gather(v) for k, v in tefo_stats.items()}

    if accelerator.is_main_process:
        sum_tefo_stats = {k: torch.sum(v) for k, v in gathered_tefo_stats.items()}

        mean_val_loss = sum_tefo_stats['loss'] / sum_tefo_stats['num_batches']
        ppl = torch.exp(sum_tefo_stats['log_probs'] / sum_tefo_stats['tokens'])
        accuracy = sum_tefo_stats['accuracy'] / sum_tefo_stats['samples']
        solved = sum_tefo_stats['solved'] / sum_tefo_stats['samples']
        mean_mse = sum_tefo_stats['mse'] / sum_tefo_stats['samples']
        valid = sum_tefo_stats['valid'] / sum_tefo_stats['samples']

        logger.info(
            f"{set_name} - Step: {step} - Mean Loss: {mean_val_loss.item():.4f}"
            f" - Mean PPL: {ppl.item():.4f}"
            f" - Mean Acc: {accuracy.item():.4f}"
            f" - Mean Solved: {solved.item():.4f}"
            f" - Mean MSE: {mean_mse.item():.4f}"
            f" - Mean Valid: {valid.item():.4f}"
            f" - Samples: {int(sum_tefo_stats['samples'].item())}"
        )

        if tb_logger is not None:
            tb_logger.add_scalar(f"{set_name}-tefo/loss", mean_val_loss.item(), step)
            tb_logger.add_scalar(f"{set_name}-tefo/ppl", ppl.item(), step)
            tb_logger.add_scalar(f"{set_name}-tefo/solved", solved.item(), step)
            tb_logger.add_scalar(f"{set_name}-tefo/accuracy", accuracy.item(), step)
            tb_logger.add_scalar(f"{set_name}-tefo/mse", mean_mse.item(), step)
            tb_logger.add_scalar(f"{set_name}-tefo/valid_eqs", valid.item(), step)
            tb_logger.flush()


    # logger.info(f"End teacher forcing evaluation for {set_name}!")



def autoregressive_evaluation(model, step, set_name, data_loader, sympy_data, cfg, accelerator, device, logger, tb_logger=None):

    logger.info(f"Start autoregressive evaluation for {set_name}!")

    model.eval()

    max_length = cfg.model.max_len
    eoe_index = sympy_data.eoe_index

    auto_stats = defaultdict(lambda: torch.tensor(0, device=device, dtype=torch.float))

    with torch.no_grad():
        for val_batch in data_loader:
            val_batch = sympy_data.batch_to_device(val_batch, device)

            pred_token = val_batch['in_equation'][:,0].unsqueeze(1)
            for _ in range(max_length):
                with torch.no_grad():
                    with accelerator.autocast():
                        logits = model(val_batch['mantissa'],
                                       val_batch['exponent'],
                                       pred_token)

                predicted_tokens = logits.argmax(dim=-1)
                pred_token = torch.concat([pred_token, predicted_tokens[:,-1].unsqueeze(1)], dim=1)
                if all([eoe_index in  s for s in pred_token]):
                    break

            pred_token = pred_token[:,1:]

            for pred_eq, truq_eq in zip(pred_token, val_batch['trg_equation']):
                if eoe_index in pred_eq:
                    eq_eoe = torch.where(pred_eq == eoe_index)[0][0]
                    pred_eq = pred_eq[:eq_eoe]
                    eq_eoe = torch.where(truq_eq == eoe_index)[0][0]
                    truq_eq = truq_eq[:eq_eoe]

                    if pred_eq.shape[0] == truq_eq.shape[0]:
                        accuracy = torch.sum(pred_eq == truq_eq) / truq_eq.shape[0]
                        solved = accuracy == 1.0
                        auto_stats['accuracy'] += accuracy
                        auto_stats['solved'] += solved
                    else:
                        min_idx = min(pred_eq.shape[0], truq_eq.shape[0])
                        max_idx = max(pred_eq.shape[0], truq_eq.shape[0])
                        accuracy = torch.sum(pred_eq[:min_idx] == truq_eq[:min_idx])/ max_idx
                        auto_stats['accuracy'] += accuracy
                        auto_stats['solved'] += 0
                    auto_stats['has_eoe'] += 1

                auto_stats['samples'] += 1

    gathered_auto_stats = {k: accelerator.gather(v) for k, v in auto_stats.items()}

    if accelerator.is_main_process:
        sum_auto_stats = {k: torch.sum(v) for k, v in gathered_auto_stats.items()}

        if 'accuracy' in sum_auto_stats:

            accuracy = sum_auto_stats['accuracy'] / sum_auto_stats['samples']
            solved = sum_auto_stats['solved'] / sum_auto_stats['samples']
            # mean_mse = sum_auto_stats['mse'] / sum_auto_stats['samples']
            # valid = sum_auto_stats['valid'] / sum_auto_stats['samples']

            logger.info(
                f"{set_name} - Step: {step} - Autoregressive Validation - "
                f" - Mean Acc: {accuracy.item():.4f}"
                f" - Mean Solved: {solved.item():.4f}"
                # f" - Mean MSE: {mean_mse.item():.4f}"
                # f" - Mean Valid: {valid.item():.4f}"
                f" - Samples: {int(sum_auto_stats['samples'].item())}"
            )

            if tb_logger is not None:
                tb_logger.add_scalar(f"{set_name}-auto/solved", solved.item(), step)
                tb_logger.add_scalar(f"{set_name}-auto/accuracy", accuracy.item(), step)
                tb_logger.flush()

    # logger.info(f"End autoregressive evaluation for {set_name}!")


def main_eval(config_dict, exp_folder, checkpoint_file):
    """
    Launch evaluation
    """
    cfg = Config(config_dict=config_dict)

    torch.set_float32_matmul_precision("medium")

    accelerator = accelerate.Accelerator(**cfg.accelerate)
    device = accelerator.device

    world_size = accelerator.num_processes
    is_rank_zero = accelerator.is_main_process
    rank = accelerator.process_index

    accelerate.utils.set_seed(cfg.train.seed)


    logger = logging.getLogger(__name__)

    if is_rank_zero:
        logging.basicConfig(
            format=f"[%(asctime)s][rank {rank}][%(levelname)s][%(name)s] - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(exp_folder / "eval_logfile.txt"),
            ],
        )


        logger.info(f"########  Accelerate: world size {world_size} - rank {rank}")

        logger.info(bold("######################################################"))
        logger.info(bold("########         START   EVALUATION         ##########"))
        logger.info(bold("######################################################"))

        logger.info(f"########  Project:    {cfg.experiment.project_name}")
        logger.info(f"########  Session:    {cfg.experiment.session_name}")
        logger.info(f"########  Experiment: {cfg.experiment.experiment_name}")

        logger.info(bold("############### CONFIGURATION"))
        cfg_dict = cfg.get_dict()
        for k, v in cfg_dict.items():
            logger.info(f"{k}: {v}")

    logger.info(bold(f"############### LOAD DATA on rank {rank}"))

    fh = logging.FileHandler(exp_folder / "info.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    sympy_data = SymPySimpleDataModule(cfg,logger=logger,)

    e_master = EquationMaster(cfg,logger=logger,)

    accelerator.wait_for_everyone()

    logger.info(bold(f"############### SETUP DATA on rank {rank}"))
    valid_loader = sympy_data.get_valid_loader()
    test_loader = sympy_data.get_test_loader()


    logger.info(bold(f"############### LOAD MODEL on rank {rank}"))
    cfg.model.trg_vocab_size = sympy_data.vocab_size
    cfg.model.seq_vocab_size = sympy_data.vocab_size
    model = GPRTransformer(cfg.model)



    model, valid_loader, test_loader = accelerator.prepare(model, valid_loader, test_loader)

    accelerator.load_state(checkpoint_file)

    if is_rank_zero:
        def count_parameters(parameters):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(
            f"#### trainable_parameters {count_parameters(model.parameters())}"
        )


    teacher_forcing_evaluation(model,0, "valid_set", valid_loader, sympy_data, e_master, accelerator, device, logger)
    teacher_forcing_evaluation(model,0, "feynman", test_loader, sympy_data, e_master, accelerator, device, logger)

    autoregressive_evaluation(model, 0,"valid_set", valid_loader, sympy_data, cfg, accelerator, device, logger)
    autoregressive_evaluation(model, 0,"feynman", test_loader, sympy_data, cfg, accelerator, device, logger)






if __name__ == "__main__":
    from functools import reduce  # forward compatibility for Python 3
    import operator


    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)


    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


    def convert_string_value(value):
        if value in ("false", "False"):
            value = False
        elif value in ("true", "True"):
            value = True
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        return value


    default_dir = "/home/joerg/workspace/experiments/first_hpo_3/setup_rel100_s20k_lr3e-4_wd01-000"
    default_dir = "/home/joerg/workspace/python/github/ScalingSymbolicRegression/.experiments/gpr_project/design_model/test_inference_wsd_cpr-000"

    parser = argparse.ArgumentParser(description="Train GPT Model")
    parser.add_argument("-d", "--dir", type=str, default=default_dir, help="config file name")
    parser.add_argument("-c", "--checkpoint", type=str, default='last', help="checkpoint file name")

    args, unknown_args = parser.parse_known_args()

    experiment_dir = args.dir
    checkpoint = args.checkpoint

    if checkpoint == 'last':
        if os.path.exists(os.path.join(experiment_dir, "pytorch_model.bin")):
            checkpoint_file = experiment_dir
        elif os.path.exists(os.path.join(experiment_dir, "states", "pytorch_model.bin")):
            checkpoint_file = os.path.join(experiment_dir, "states")
        else:
            raise FileNotFoundError("No last checkpoint found")
    else:
        checkpoint_file = os.path.join(experiment_dir, checkpoint)

    config_file = os.path.join(experiment_dir, "config.yml")
    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    for arg in unknown_args:
        if "=" in arg:
            keys = arg.split("=")[0].split(".")
            value = convert_string_value(arg.split("=")[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")

    experiment_dir = pathlib.Path(experiment_dir)

    main_eval(config_dict, experiment_dir, checkpoint_file)
