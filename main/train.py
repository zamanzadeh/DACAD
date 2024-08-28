import sys
sys.path.append("..")
import os
import numpy as np
import random
import torch
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.util_progress_log import ProgressMeter, get_logger, get_dataset_type
import json
from argparse import ArgumentParser
from algorithms import get_algorithm


def main(args):
    # Torch RNG
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Python RNG
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # First configure our logger
    log = get_logger(
        os.path.join(args.experiments_main_folder, args.experiment_folder, str(args.id_src) + "-" + str(args.id_trg),
                     args.log))

    # Some functions and variables for logging
    dataset_type = get_dataset_type(args)

    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type == "smd":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "msl":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "boiler":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_val_iteration = args.num_val_iteration

    # LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)
    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    dataset_val_src = get_dataset(args, domain_type="source", split_type="val")

    dataset_trg = get_dataset(args, domain_type="target", split_type="train")
    dataset_val_trg = get_dataset(args, domain_type="target", split_type="val")

    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=True)
    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)

    dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)

    max_num_val_iteration = min(len(dataloader_val_src), len(dataloader_val_trg))
    if max_num_val_iteration < num_val_iteration:
        num_val_iteration = max_num_val_iteration

    # Calculate input_channels_dim and input_static_dim
    input_channels_dim = dataset_src[0]['sequence'].shape[1]
    input_static_dim = dataset_src[0]['static'].shape[0] if 'static' in dataset_src[0] else 0

    # Get our algorithm
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim)

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(args.id_trg))

    # Initialize progress metrics before training
    count_step = 0
    best_val_score = -100

    src_mean, src_std = dataset_src.get_statistic()
    trg_mean, trg_std = dataset_src.get_statistic()
    for i in range(args.num_epochs):
        dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
        dataloader_iterator = iter(dataloader_trg)

        for i_batch, sample_batched_src in enumerate(dataloader_src):
                sample_batched_src = sample_batched_src
                for key, value in sample_batched_src.items():
                    sample_batched_src[key] = sample_batched_src[key]#.to(device= 'cuda', non_blocking= True)
                # Current model does not support smaller batches than batch_size (due to queue ptr)
                if len(sample_batched_src['sequence']) != batch_size:
                    continue

                try:
                    sample_batched_trg = next(dataloader_iterator)
                except StopIteration:
                    dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                                shuffle=True, num_workers=0, drop_last=True)
                    dataloader_iterator = iter(dataloader_trg)
                    sample_batched_trg = next(dataloader_iterator)

                for key, value in sample_batched_trg.items():
                    sample_batched_trg[key] = sample_batched_trg[key]#.to(device= 'cuda', non_blocking= True)

                # Current model does not support smaller batches than batch_size (due to queue ptr)
                if len(sample_batched_trg['sequence']) != batch_size:
                    continue

                # Training step of algorithm
                algorithm.step(sample_batched_src, sample_batched_trg, count_step=count_step, epoch=i,
                               src_mean=src_mean, src_std=src_std, trg_mean=trg_mean, trg_std=trg_std)

                count_step += 1
                if count_step % len(dataloader_src) == 0:
                    progress = ProgressMeter(
                        len(dataloader_src),
                        algorithm.return_metrics(),
                        prefix="Epoch: [{}]".format(i))

                    log(progress.display(i_batch + 1, is_logged=True))

                    # Refresh the saved metrics for algorithm
                    algorithm.init_metrics()

                    # Refresh the validation meters of algorithm
                    algorithm.init_pred_meters_val()

                    # turn algorithm into eval mode
                    algorithm.eval()

                    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                                    shuffle=True, num_workers=0, drop_last=True)
                    dataloader_val_src_iterator = iter(dataloader_val_src)

                    dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                                    shuffle=True, num_workers=0, drop_last=True)
                    dataloader_val_trg_iterator = iter(dataloader_val_trg)

                    for i_batch_val in range(num_val_iteration):
                        sample_batched_val_src = next(dataloader_val_src_iterator)
                        sample_batched_val_trg = next(dataloader_val_trg_iterator)

                        # Validation step of algorithm
                        algorithm.step(sample_batched_val_src, sample_batched_val_trg, count_step=count_step,
                                       src_mean=src_mean, src_std=src_std, trg_mean=trg_mean, trg_std=trg_std)

                    progress_val = ProgressMeter(
                        num_val_iteration,
                        algorithm.return_metrics(),
                        prefix="Epoch: [{}]".format(i))

                    metrics_pred_val_src = algorithm.pred_meter_val_src.get_metrics()
                    metrics_pred_val_trg = algorithm.pred_meter_val_trg.get_metrics()

                    log("VALIDATION RESULTS")
                    log(progress_val.display(i_batch_val + 1, is_logged=True))

                    log("VALIDATION SOURCE PREDICTIONS")
                    log_scores(args, dataset_type, metrics_pred_val_src)

                    if dataset_type == "msl":
                        cur_val_score = metrics_pred_val_src["best_f1"]
                    elif dataset_type == "boiler":
                        cur_val_score = metrics_pred_val_src["best_f1"]
                    elif dataset_type == "smd":
                        cur_val_score = metrics_pred_val_trg["best_f1"]
                    else:
                        cur_val_score = metrics_pred_val_src["mac_f1"]

                    if cur_val_score > best_val_score:
                        algorithm.save_state(experiment_folder_path)

                        best_val_score = cur_val_score
                    # algorithm.save_state(experiment_folder_path)
                    log("VALIDATION TARGET PREDICTIONS")
                    log_scores(args, dataset_type, metrics_pred_val_trg)

                    # turn algorithm into training mode
                    algorithm.train()

                    # Refresh the saved metrics for algorithm
                    algorithm.init_metrics()

                else:
                    continue
                break


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='dacad')

    parser.add_argument('-dr', '--dropout', type=float, default=0.1)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99)  # DACAD
    parser.add_argument('-qs', '--queue_size', type=int, default=98304)  # DACAD
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true')  # DACAD
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)  # 2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=200)  # 2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64')  # All TCN models
    parser.add_argument('--kernel_size_TCN', type=int, default=3)  # All TCN models
    parser.add_argument('--dilation_factor_TCN', type=int, default=2)  # All TCN models
    parser.add_argument('--stride_TCN', type=int, default=1)  # All TCN models
    parser.add_argument('--hidden_dim_MLP', type=int, default=256)  # All classifier and discriminators

    # The weight of the domain classification loss
    parser.add_argument('-w_d', '--weight_domain', type=float, default=0.1)
    # Below weights are defined for DACAD
    parser.add_argument('--weight_loss_src', type=float, default=0.0)
    parser.add_argument('--weight_loss_trg', type=float, default=0.0)
    parser.add_argument('--weight_loss_ts', type=float, default=0.0)
    parser.add_argument('--weight_loss_disc', type=float, default=0.5)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)
    parser.add_argument('--weight_loss_src_sup', type=float, default=0.1)
    parser.add_argument('--weight_loss_trg_inj', type=float, default=0.1)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='smd')

    parser.add_argument('--path_src', type=str, default='../datasets/MSL_SMAP') #../datasets/Boiler/   ../datasets/MSL_SMAP
    parser.add_argument('--path_trg', type=str, default='../datasets/MSL_SMAP') #../datasets/SMD/test
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=str, default='1-5')
    parser.add_argument('--id_trg', type=str, default='1-1')

    parser.add_argument('--task', type=str, default='decompensation')

    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    if not os.path.exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.makedirs(os.path.join(args.experiments_main_folder, args.experiment_folder), exist_ok=True)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder,
                                       str(args.id_src) + "-" + str(args.id_trg))):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder,
                              str(args.id_src) + "-" + str(args.id_trg)))

    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)