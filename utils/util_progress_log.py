import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, \
    precision_recall_curve


def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log


# We keep the dictionary of metrics for each dataset and task
dict_metrics = {"smd": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                "msl": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                "boiler": {"acc": accuracy_score, "mac_f1": f1_score, "w_f1": f1_score},
                }


def get_dataset_type(args):
    if "SMD" in args.path_src:
        return "smd"
    elif "MSL" in args.path_src:
        return "msl"
    elif "Boiler" in args.path_src:
        return "boiler"
    else:
        return "sensor"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, is_logged=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # If the progress will be logged (instead of printing)
        # we will return the string so that it can be logged in the main script.
        if not is_logged:
            print('\t'.join(entries))
        else:
            return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class PredictionMeter(object):
    def __init__(self, args):
        self.args = args
        self.dataset_type = get_dataset_type(args)
        self.target_list = []
        self.output_list = []
        self.id_patient_list = []
        self.stay_hours_list = []

        # Initialize the metric dictionary to be returned
        self.dict_metrics = {}
        self.dict_metrics = dict_metrics[self.dataset_type]

    def update(self, target, output, id_patient=None, stay_hour=None):
        if self.dataset_type == "smd":
            output_np = output.detach().cpu().numpy().flatten()
        elif self.dataset_type == "msl":
            output_np = output.detach().cpu().numpy().flatten()
        elif self.dataset_type == "boiler":
            output_np = output.detach().cpu().numpy().flatten()
        else:
            output_np = output.detach().cpu().numpy().argmax(axis=1).flatten()
        target_np = target.detach().cpu().numpy().flatten()

        self.output_list = self.output_list + list(output_np)
        self.target_list = self.target_list + list(target_np)

        # Below is especially helpful for saving the predictions in eval mode
        if id_patient is not None:
            id_patient_np = id_patient.numpy().flatten()
            self.id_patient_list = self.id_patient_list + list(id_patient_np)

        if stay_hour is not None:
            stay_hour_np = stay_hour.numpy().flatten()
            self.stay_hours_list = self.stay_hours_list + list(stay_hour_np)

    def get_metrics(self):
        return_dict = {}
        output = np.array(self.output_list)
        target = np.array(self.target_list)

        avg_prc = average_precision_score(target, output, pos_label=1)
        roc_auc =0
        try:
            roc_auc = roc_auc_score(target, output)
        except ValueError:
            pass
        prec, rec, thr = precision_recall_curve(target, output, pos_label=1)
        prec = np.where(prec == np.nan, 0.0, prec)
        rec = np.where(rec == np.nan, 0.0, rec)

        with np.errstate(invalid='ignore'):
            f1score = np.where((rec + prec) == 0.0, 0.0, 2 * prec * rec / (prec + rec))
        best_f1_index = np.argmax(f1score)
        return_dict["best_f1"] = f1score[best_f1_index]
        return_dict["best_prec"] = prec[best_f1_index]
        return_dict["best_rec"] = rec[best_f1_index]
        return_dict["best_thr"] = thr[best_f1_index]
        return_dict["avg_prc"] = avg_prc
        return_dict["roc_auc"] = roc_auc
        output = np.where(output[:] > thr[best_f1_index], 1, 0)
        return_dict["macro_F1"] = f1_score(target, output, average="macro")
        return return_dict

        """
        if self.task != "los":
            roc_auc = roc_auc_score(target, output)
            avg_prc = average_precision_score(target, output)

            return_dict["roc_auc"] = roc_auc
            return_dict["avg_prc"] = avg_prc

        else:
            kappa = cohen_kappa_score(output, target)
            return_dict["kappa"] = kappa
        """

        """
        for k in self.dict_metrics:
            output = np.where(output[:] > 0.9, 1, 0)
            if k == "mac_f1":
                return_dict[k] = self.dict_metrics[k](target, output, average="macro")
            elif k == "w_f1":
                return_dict[k] = self.dict_metrics[k](target, output, average="weighted")
            else:
                return_dict[k] = self.dict_metrics[k](target, output)

        return return_dict
        """


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def write_to_tensorboard(writer, progress, count_step, split_type="train", task="decompensation", auroc_s=None,
                         auprc_s=None, auroc_t=None, auprc_t=None, kappa_s=None, kappa_t=None):
    if len(progress.meters) == 11:
        writer.add_scalar('Loss_Total/' + split_type, progress.meters[10].avg, count_step)
        writer.add_scalar('Loss_Source_CL/' + split_type, progress.meters[2].avg, count_step)
        writer.add_scalar('Loss_Target_CL/' + split_type, progress.meters[4].avg, count_step)
        writer.add_scalar('Loss_SourTar_CL/' + split_type, progress.meters[6].avg, count_step)
        writer.add_scalar('Loss_Source_Pred/' + split_type, progress.meters[8].avg, count_step)

        writer.add_scalar('Acc@1_Source_CL/' + split_type, progress.meters[3].avg, count_step)
        writer.add_scalar('Acc@1_Target_CL/' + split_type, progress.meters[5].avg, count_step)
        writer.add_scalar('Acc@1_SourTar_CL/' + split_type, progress.meters[7].avg, count_step)

        if task != "los":
            if split_type == "train":
                writer.add_scalar('AUROC_Source_Pred/' + split_type, progress.meters[9].avg, count_step)

            if split_type == "val":
                writer.add_scalar('AUROC_Source_Pred/' + split_type, auroc_s, count_step)
                writer.add_scalar('AUPRC_Source_Pred/' + split_type, auprc_s, count_step)
                writer.add_scalar('AUROC_Target_Pred/' + split_type, auroc_t, count_step)
                writer.add_scalar('AUPRC_Target_Pred/' + split_type, auprc_t, count_step)

        else:
            if split_type == "train":
                writer.add_scalar('KAPPA_Source_Pred/' + split_type, progress.meters[9].avg, count_step)

            if split_type == "val":
                writer.add_scalar('KAPPA_Source_Pred/' + split_type, kappa_s, count_step)
                writer.add_scalar('KAPPA_Target_Pred/' + split_type, kappa_t, count_step)

    elif len(progress.meters) == 13:
        writer.add_scalar('Loss_Total/' + split_type, progress.meters[12].avg, count_step)
        writer.add_scalar('Loss_Source_CL/' + split_type, progress.meters[2].avg, count_step)
        writer.add_scalar('Loss_Target_CL/' + split_type, progress.meters[4].avg, count_step)
        writer.add_scalar('Loss_SourTar_CL/' + split_type, progress.meters[6].avg, count_step)
        writer.add_scalar('Loss_SourTar_Disc/' + split_type, progress.meters[8].avg, count_step)
        writer.add_scalar('Loss_Source_Pred/' + split_type, progress.meters[10].avg, count_step)

        writer.add_scalar('Acc@1_Source_CL/' + split_type, progress.meters[3].avg, count_step)
        writer.add_scalar('Acc@1_Target_CL/' + split_type, progress.meters[5].avg, count_step)
        writer.add_scalar('Acc@1_SourTar_CL/' + split_type, progress.meters[7].avg, count_step)
        writer.add_scalar('Acc@1_SourTar_Disc/' + split_type, progress.meters[7].avg, count_step)

        if task != "los":
            if split_type == "train":
                writer.add_scalar('AUROC_Source_Pred/' + split_type, progress.meters[11].avg, count_step)

            if split_type == "val":
                writer.add_scalar('AUROC_Source_Pred/' + split_type, auroc_s, count_step)
                writer.add_scalar('AUPRC_Source_Pred/' + split_type, auprc_s, count_step)
                writer.add_scalar('AUROC_Target_Pred/' + split_type, auroc_t, count_step)
                writer.add_scalar('AUPRC_Target_Pred/' + split_type, auprc_t, count_step)

        else:
            if split_type == "train":
                writer.add_scalar('KAPPA_Source_Pred/' + split_type, progress.meters[11].avg, count_step)

            if split_type == "val":
                writer.add_scalar('KAPPA_Source_Pred/' + split_type, kappa_s, count_step)
                writer.add_scalar('KAPPA_Target_Pred/' + split_type, kappa_t, count_step)


