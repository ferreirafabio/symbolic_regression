import os
import sys
import socket
import argparse
import collections
import yaml, tqdm
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import torch.cuda
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import accelerate

# from gpr.data.generators import PolynomialGenerator
from gpr.data.loaders import SymPySimpleDataModule
from gpr.model.gpr_transformer import GPRTransformer
from gpr.utils.configuration import Config
from gpr.utils.folder_manager import get_experiment_folder
from gpr.model.module.ce_loss import FlashCrossEntropyLoss


def bold(msg):
    return f"\033[1m{msg}\033[0m"


def main(config_dict):
    """
    Launch pretraining
    """
    cfg = Config(config_dict=config_dict)

    torch.set_float32_matmul_precision("medium")

    accelerator = accelerate.Accelerator(**cfg.accelerate)
    device = accelerator.device

    world_size = accelerator.num_processes
    is_rank_zero = accelerator.is_main_process
    rank = accelerator.process_index

    accelerate.utils.set_seed(cfg.train.seed)

    if is_rank_zero:
        exp_folder = get_experiment_folder(**cfg.experiment, new_folder=is_rank_zero)

    logger = logging.getLogger(__name__)

    if is_rank_zero:
        logging.basicConfig(
            format=f"[%(asctime)s][rank {rank}][%(levelname)s][%(name)s] - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(exp_folder / "logfile.txt"),
            ],
        )

        logger.info(f"#### Load logger on rank {rank}")
        tb_logger = SummaryWriter(
            log_dir=exp_folder,
        )

        cfg.save_config(exp_folder)
        fh = logging.FileHandler(exp_folder / "info.log")
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        logger.info(f"########  Accelerate: world size {world_size} - rank {rank}")

        logger.info(bold("######################################################"))
        logger.info(bold("########          START   TRAINING          ##########"))
        logger.info(bold("######################################################"))

        logger.info(f"########  Project:    {cfg.experiment.project_name}")
        logger.info(f"########  Session:    {cfg.experiment.session_name}")
        logger.info(f"########  Experiment: {cfg.experiment.experiment_name}")
        logger.info(f"save logs and checkpoints in: {exp_folder.as_posix()}")

        logger.info(bold("############### CONFIGURATION"))
        cfg_dict = cfg.get_dict()
        for k, v in cfg_dict.items():
            logger.info(f"{k}: {v}")

    logger.info(bold(f"############### LOAD DATA on rank {rank}"))

    # Instantiate the equation generator
    sympy_data = SymPySimpleDataModule(config_path=config_file,
                                       logger=logger,
                                       )
    accelerator.wait_for_everyone()

    logger.info(bold(f"############### SETUP DATA on rank {rank}"))
    valid_loader = sympy_data.get_valid_loader()
    # local_rank=accelerator.local_process_index

    logger.info(bold(f"############### LOAD MODEL on rank {rank}"))

    cfg.model.trg_vocab_size = sympy_data.vocab_size
    cfg.model.seq_vocab_size = sympy_data.vocab_size

    model = GPRTransformer(cfg.model)


    model, valid_loader = accelerator.prepare(model, valid_loader)

    if is_rank_zero:
        def count_parameters(parameters):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(
            f"#### trainable_parameters {count_parameters(model.parameters())}"
        )

    loss_func = FlashCrossEntropyLoss(ignore_index=sympy_data.ignore_index,
                                      reduction='mean')

    if cfg.train.max_steps is None and cfg.train.max_epochs is None:
        raise ValueError("You must specify either max_steps or max_epochs")
    if cfg.train.max_steps is None:
        cfg.train.max_steps = 1e15
        logger.info(f"Start training for {cfg.train.max_epochs} epochs")
    if cfg.train.max_epochs is None:
        cfg.train.max_epochs = 1e15
        logger.info(f"Start training for {cfg.train.max_steps} steps")


    for val_batch in valid_loader:
        val_batch = sympy_data.batch_to_device(val_batch, device)
        with accelerator.autocast():
            logits = model(val_batch['mantissa'],
                           val_batch['exponent'],
                           val_batch['in_equation'])
            val_loss = loss_func(logits.view(-1, logits.size(-1)), val_batch['trg_equation'].view(-1))

        sample_count = val_batch['trg_len'].size(0)
        token_count = torch.sum(val_batch['trg_len'], dtype=torch.float)

        log_probs = val_loss * token_count
        predicted_tokens = logits.argmax(dim=-1)
        # logger.info(f"logits shape: {logits.shape}")
        # logger.info(f"predicted_tokens shape: {predicted_tokens.shape}")
        # logger.info(f"trq_seq shape: {trg_seq.shape}")
        # logger.info(f"predicted_tokens view shape: {predicted_tokens.view(-1).shape}")
        # logger.info(f"trq_seq view: {trg_seq.view(-1).shape}")

        correct_tokens = predicted_tokens == val_batch['trg_equation']
        ignore_mask = val_batch['trg_equation'] != sympy_data.ignore_index

        batch_accuracy = torch.sum(correct_tokens & ignore_mask, dim=-1) / val_batch['trg_len']
        batch_solved = torch.sum(batch_accuracy == 1.0)

        acc_loss += val_loss
        acc_log_probs += log_probs
        acc_accuracy += torch.sum(batch_accuracy)
        acc_solved += torch.sum(batch_solved)

        acc_tokens += token_count
        acc_samples += sample_count
        acc_num_batches += 1

        # compute MSE between predicted and true equation of the current val batch
        true_strs = sympy_data.indices_to_string(val_batch['trg_equation'])
        pred_strs = sympy_data.indices_to_string(predicted_tokens)

        batch_sum_mse, batch_sum_valid_eq = sympy_data.compute_mse(pred_strs, true_strs)
        acc_equation_mse += batch_sum_mse
        acc_valid_equation += batch_sum_valid_eq

    gathered_val_loss = accelerator.gather(acc_loss)
    gathered_num_batches = accelerator.gather(acc_num_batches)
    gathered_acc_log_probs = accelerator.gather(acc_log_probs)
    gathered_acc_accuracy = accelerator.gather(acc_accuracy)
    gathered_acc_solved = accelerator.gather(acc_solved)
    gathered_mse = accelerator.gather(acc_equation_mse)
    gathered_valid = accelerator.gather(acc_valid_equation)
    gathered_acc_samples = accelerator.gather(acc_samples)
    gathered_acc_tokens = accelerator.gather(acc_tokens)

    if is_rank_zero:
        acc_loss = torch.sum(gathered_val_loss)
        num_batches = torch.sum(gathered_num_batches)
        acc_log_probs = torch.sum(gathered_acc_log_probs)
        acc_accuracy = torch.sum(gathered_acc_accuracy)
        acc_solved = torch.sum(gathered_acc_solved)
        acc_mse = torch.sum(gathered_mse)
        acc_valid = torch.sum(gathered_valid)
        acc_samples = torch.sum(gathered_acc_samples)
        acc_tokens = torch.sum(gathered_acc_tokens)

        mean_val_loss = acc_loss / num_batches
        ppl = torch.exp(acc_log_probs / acc_tokens)
        accuracy = acc_accuracy / acc_samples
        solved = acc_solved / acc_samples
        mean_mse = acc_mse / acc_samples
        valid = acc_valid / acc_samples

        logger.info(
            f"Validation at step {step} - Mean Loss: {mean_val_loss.item():.4f}"
            f" - Mean PPL: {ppl.item():.4f}"
            f" - Mean Acc: {accuracy.item():.4f}"
            f" - Mean Solved: {solved.item():.4f}"
            f" - Mean MSE: {mean_mse.item():.4f}"
            f" - Mean Valid: {valid.item():.4f}"
            f" - Samples: {int(acc_samples.item())}"
        )
        tb_logger.add_scalar(f"valid/loss", mean_val_loss.item(), step)
        tb_logger.add_scalar(f"valid/ppl", ppl.item(), step)
        tb_logger.add_scalar("valid/solved", solved.item(), step)
        tb_logger.add_scalar(f"valid/accuracy", accuracy.item(), step)
        tb_logger.add_scalar(f"valid/mse", mean_mse.item(), step)
        tb_logger.add_scalar(f"valid/valid_eqs", valid.item(), step)
        tb_logger.flush()


    logger.info("End evaluation!")


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


    default_config_name = "default_config.yaml"

    parser = argparse.ArgumentParser(description="Train GPT Model")
    parser.add_argument(
        "-c", "--config", type=str, default=default_config_name, help="config file name"
    )

    args, unknown_args = parser.parse_known_args()

    config_name = args.config
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_file = os.path.join("config", config_name)
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

    main(config_dict)
