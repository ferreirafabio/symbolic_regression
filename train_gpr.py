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

#from gpr.data.generators import PolynomialGenerator
from gpr.data.loaders import SymPySimpleDataModule
from gpr.model.gpr_transformer import GPRTransformer
from gpr.utils.configuration import Config
from gpr.utils.folder_manager import get_experiment_folder
from gpr.train.configure_optimizer import configure_optimizer
from gpr.model.module.ce_loss import FlashCrossEntropyLoss
from gpr.data.equation_master import EquationMaster
from infer_gpr import teacher_forcing_evaluation, autoregressive_evaluation

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
    sympy_data = SymPySimpleDataModule(cfg, logger=logger, )
    e_master = EquationMaster(cfg, logger=logger, )

    accelerator.wait_for_everyone()

    logger.info(bold(f"############### SETUP DATA on rank {rank}"))
    train_loader = sympy_data.get_train_loader()
    valid_loader = sympy_data.get_valid_loader()
    test_loader = sympy_data.get_test_loader()
    # local_rank=accelerator.local_process_index

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Valid loader: {len(valid_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")



    logger.info(bold(f"############### LOAD MODEL on rank {rank}"))


    cfg.model.trg_vocab_size = sympy_data.vocab_size
    cfg.model.seq_vocab_size = sympy_data.vocab_size

    model = GPRTransformer(cfg.model)


    # optimizer, lr_scheduler = configure_optimizer(cfg.optim, model, logger)
    loss_func = FlashCrossEntropyLoss(ignore_index=sympy_data.ignore_index, reduction='mean')

    model = accelerator.prepare(model)


    if is_rank_zero:

        def count_parameters(parameters):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(
            f"#### trainable_parameters {count_parameters(model.parameters())}"
        )


    optimizer, lr_scheduler = configure_optimizer(cfg.optim, model, logger)

    if cfg.train.max_steps is None and cfg.train.max_epochs is None:
        raise ValueError("You must specify either max_steps or max_epochs")
    if cfg.train.max_steps is None:
        cfg.train.max_steps = 1e15
        logger.info(f"Start training for {cfg.train.max_epochs} epochs")
    if cfg.train.max_epochs is None:
        cfg.train.max_epochs = 1e15
        logger.info(f"Start training for {cfg.train.max_steps} steps")

    optimizer, train_dl, valid_dl = accelerator.prepare(optimizer, train_loader, valid_loader)
    RUN_TRAIN = True

    epoch = 0
    step = 0
    train_losses = []

    while RUN_TRAIN:
        logger.info(f"Start epoch {epoch}")

        model.train()
        train_acc = 0
        train_solved = 0
        train_samples = 0

        for train_batch in train_dl:

            if not RUN_TRAIN:
                break

            if step % cfg.train.val_interval == 0:

                teacher_forcing_evaluation(model, step,"valid", valid_loader, sympy_data, e_master, accelerator, device,
                                           logger)
                teacher_forcing_evaluation(model, step,"feynman", test_loader, sympy_data, e_master, accelerator, device,
                                           logger)

                # autoregressive_evaluation(model, step,"valid", valid_loader, sympy_data, cfg, accelerator, device,
                #                           logger)
                autoregressive_evaluation(model, step,"feynman", test_loader, sympy_data, cfg, accelerator, device, logger)

                # model.eval()
                # with torch.no_grad():
                #     acc_loss = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_num_batches = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_accuracy = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_solved = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_log_probs = torch.tensor(0, device=device, dtype=torch.float)
                #
                #     acc_equation_mse = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_valid_equation = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_samples = torch.tensor(0, device=device, dtype=torch.float)
                #     acc_tokens = torch.tensor(0, device=device, dtype=torch.float)
                #     val_pred_true_equation_mse = torch.tensor(0, device=device, dtype=torch.float)
                #
                #     for val_batch in valid_dl:
                #         val_batch = sympy_data.batch_to_device(val_batch, device)
                #         with accelerator.autocast():
                #             logits = model(val_batch['mantissa'],
                #                            val_batch['exponent'],
                #                            val_batch['in_equation'])
                #             val_loss = loss_func(logits.view(-1, logits.size(-1)), val_batch['trg_equation'].view(-1))
                #
                #
                #         sample_count = val_batch['trg_len'].size(0)
                #         token_count = torch.sum(val_batch['trg_len'], dtype=torch.float)
                #
                #         log_probs = val_loss * token_count
                #         predicted_tokens = logits.argmax(dim=-1)
                #         # logger.info(f"logits shape: {logits.shape}")
                #         # logger.info(f"predicted_tokens shape: {predicted_tokens.shape}")
                #         # logger.info(f"trq_seq shape: {trg_seq.shape}")
                #         # logger.info(f"predicted_tokens view shape: {predicted_tokens.view(-1).shape}")
                #         # logger.info(f"trq_seq view: {trg_seq.view(-1).shape}")
                #
                #         correct_tokens = predicted_tokens == val_batch['trg_equation']
                #         ignore_mask = val_batch['trg_equation'] != sympy_data.ignore_index
                #
                #         batch_accuracy = torch.sum(correct_tokens & ignore_mask, dim=-1) / val_batch['trg_len']
                #         batch_solved = torch.sum(batch_accuracy == 1.0)
                #
                #         acc_loss += val_loss
                #         acc_log_probs += log_probs
                #         acc_accuracy += torch.sum(batch_accuracy)
                #         acc_solved += torch.sum(batch_solved)
                #
                #         acc_tokens += token_count
                #         acc_samples += sample_count
                #         acc_num_batches += 1
                #
                #         # compute MSE between predicted and true equation of the current val batch
                #         # true_strs = sympy_data.indices_to_string(val_batch['trg_equation'])
                #         # pred_strs = sympy_data.indices_to_string(predicted_tokens)
                #         #
                #         # batch_sum_mse, batch_sum_valid_eq = sympy_data.compute_mse(pred_strs, true_strs)
                #         # acc_equation_mse += batch_sum_mse
                #         # acc_valid_equation += batch_sum_valid_eq
                #
                #     gathered_val_loss = accelerator.gather(acc_loss)
                #     gathered_num_batches = accelerator.gather(acc_num_batches)
                #     gathered_acc_log_probs = accelerator.gather(acc_log_probs)
                #     gathered_acc_accuracy = accelerator.gather(acc_accuracy)
                #     gathered_acc_solved = accelerator.gather(acc_solved)
                #     # gathered_mse = accelerator.gather(acc_equation_mse)
                #     # gathered_valid = accelerator.gather(acc_valid_equation)
                #     gathered_acc_samples = accelerator.gather(acc_samples)
                #     gathered_acc_tokens = accelerator.gather(acc_tokens)
                #
                #     if is_rank_zero:
                #
                #         acc_loss = torch.sum(gathered_val_loss)
                #         num_batches = torch.sum(gathered_num_batches)
                #         acc_log_probs = torch.sum(gathered_acc_log_probs)
                #         acc_accuracy = torch.sum(gathered_acc_accuracy)
                #         acc_solved = torch.sum(gathered_acc_solved)
                #         # acc_mse = torch.sum(gathered_mse)
                #         # acc_valid = torch.sum(gathered_valid)
                #         acc_samples = torch.sum(gathered_acc_samples)
                #         acc_tokens = torch.sum(gathered_acc_tokens)
                #
                #         mean_val_loss = acc_loss / num_batches
                #         ppl = torch.exp(acc_log_probs / acc_tokens)
                #         accuracy = acc_accuracy / acc_samples
                #         solved = acc_solved / acc_samples
                #         # mean_mse = acc_mse / acc_samples
                #         # valid = acc_valid / acc_samples
                #
                #         logger.info(
                #             f"Validation at step {step} - Mean Loss: {mean_val_loss.item():.4f}"
                #             f" - Mean PPL: {ppl.item():.4f}"
                #             f" - Mean Acc: {accuracy.item():.4f}"
                #             f" - Mean Solved: {solved.item():.4f}"
                #             # f" - Mean MSE: {mean_mse.item():.4f}"
                #             # f" - Mean Valid: {valid.item():.4f}"
                #             f" - Samples: {int(acc_samples.item())}"
                #         )
                #         tb_logger.add_scalar(f"valid/loss", mean_val_loss.item(), step)
                #         tb_logger.add_scalar(f"valid/ppl", ppl.item(), step)
                #         tb_logger.add_scalar("valid/solved", solved.item(), step)
                #         tb_logger.add_scalar(f"valid/accuracy", accuracy.item(), step)
                #         # tb_logger.add_scalar(f"valid/mse", mean_mse.item(), step)
                #         # tb_logger.add_scalar(f"valid/valid_eqs", valid.item(), step)
                #         tb_logger.flush()
                model.train()

            train_batch = sympy_data.batch_to_device(train_batch, device)
            with accelerator.autocast():

                logits = model(train_batch['mantissa'],
                               train_batch['exponent'],
                               train_batch['in_equation'])
                loss = loss_func(logits.view(-1, logits.size(-1)), train_batch['trg_equation'].view(-1))
            train_losses.append(loss.item())


            pred_tokens = logits.argmax(dim=-1)
            # batch_acc = 0
            for i in range(train_batch['trg_len'].shape[0]):
                acc = (pred_tokens[i, :train_batch['trg_len'][i]] == train_batch['trg_equation'][i, :train_batch['trg_len'][i]]).sum() / train_batch['trg_len'][i]
                train_acc += acc
                train_solved += acc == 1
                train_samples += 1

            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), cfg.train.clip_value
                )
            optimizer.step()
            lr_scheduler.step()

            gathered_loss = accelerator.gather(loss)
            if step % cfg.train.log_interval == 0 and is_rank_zero:
                mean_loss = torch.mean(gathered_loss).item()

                logger.info(f"Step {step} - Loss: {mean_loss:.4f} - Acc: {(train_acc/train_samples).item():.4f} - Solved: {(train_solved/train_samples).item():.4f}")
                tb_logger.add_scalar("train/loss", mean_loss, step)
                tb_logger.add_scalar("train/acc", (train_acc/train_samples).item(), step)
                tb_logger.add_scalar("train/solved", (train_solved/train_samples).item(), step)
                tb_logger.add_scalar("train/epoch", epoch, step)
                tb_logger.add_scalar("train/batch_size", train_batch['trg_len'].shape[0], step)
                tb_logger.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], step)
                train_acc, train_solved, train_samples = 0, 0, 0

            if step % cfg.train.log_param_interval == 0 and is_rank_zero:
                stats = {}
                for k, v in model.named_parameters():
                    v_detached = v.detach()

                    if torch.isnan(v_detached).sum() > 0:
                        logger.info(f"# NaN in param {k}")
                    if torch.isinf(v_detached).sum() > 0:
                        logger.info(f"# Inf in param {k}")

                    stats[f"param/{k}/mean"] = v_detached.mean().item()
                    if v_detached.shape[0] > 1:
                        stats[f"param/{k}/std"] = v_detached.std().item()
                        stats[f"param/{k}/min"] = v_detached.min().item()
                        stats[f"param/{k}/max"] = v_detached.max().item()
                        stats[f"param/{k}/l2m"] = (v_detached**2).mean().item()

                    if v_detached.grad is not None:
                        stats[f"grad/{k}/mean"] = v_detached.grad.mean().item()
                        stats[f"grad/{k}/std"] = v_detached.grad.std().item()
                        stats[f"grad/{k}/min"] = v_detached.grad.min().item()
                        stats[f"grad/{k}/max"] = v_detached.grad.max().item()

                for k, v in stats.items():
                    tb_logger.add_scalar(k, v, step)

                tb_logger.flush()

            accelerator.wait_for_everyone()

            if step % cfg.train.save_interval == 0:
                if is_rank_zero:
                    accelerator.save_state(exp_folder / "states", safe_serialization=False)
                    torch.save(
                        {"step": step, "epoch": epoch}, exp_folder / "current_step.pt"
                    )

            if step > cfg.train.max_steps or epoch > cfg.train.max_epochs:
                RUN_TRAIN = False
                break
            step += 1

        logger.info(f"### End epoch {epoch}")
        logger.info(f"### Mean Train Loss: {np.mean(train_losses)}")
        train_losses = []
        epoch += 1

    accelerator.wait_for_everyone()
    if is_rank_zero:
        accelerator.save_model(model, exp_folder, safe_serialization=False)

    logger.info(f"Finished saving model weights on rank {rank}")
    logger.info("End training!")


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
