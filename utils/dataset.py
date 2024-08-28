import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.augmentations import Injector


def get_dataset(args, domain_type, split_type):
    """
    Return the correct dataset object that will be fed into datalaoder
    args: args of main script
    domain_type: "source" or "target"
    split_type: "train" or "val" or "test"
    """
    
    if "SMD" in args.path_src:
        if domain_type == "source":
            return SMDDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return SMDDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

    elif "MSL" in args.path_src:
        if domain_type == "source":
            return MSLDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return MSLDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

    elif "Boiler" in args.path_src:
        if domain_type == "source":
            return BoilerDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return BoilerDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

class MSLDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        # self.mean = None
        # self.std = None
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        with open(os.path.join(self.root_dir, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')

        # data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
        data_info = csv_reader[csv_reader['chan_id'] == self.subject_id]

        path_sequence = os.path.join(self.root_dir, 'test/', str(self.subject_id) + ".npy")
        temp = np.load(path_sequence)
        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean = np.mean(temp, axis=0)
        self.std = np.std(temp, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (temp - self.mean) / self.std

        labels = []
        for index, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.label = np.asarray(labels)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class MSLDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative
        # self.mean = None
        # self.std = None
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        with open(os.path.join(self.root_dir, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')

        # data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
        data_info = csv_reader[csv_reader['chan_id'] == self.subject_id]

        path_sequence = os.path.join(self.root_dir, 'test/', str(self.subject_id) + ".npy")
        temp = np.load(path_sequence)
        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean = np.mean(temp, axis=0)
        self.std = np.std(temp, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (temp - self.mean) / self.std

        labels = []
        for index, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.label = np.asarray(labels)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0] - w_size) / stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st + w_size]
            if self.label[st:st + w_size].any() > 0:
                lbl = 1
            else:
                lbl = 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class SMDDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=False, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).to(torch.float32)
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).to(torch.float32)
            negative = torch.Tensor(negative).to(torch.float32)
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, "machine-" + str(self.subject_id) + ".txt")
        self.sequence = np.loadtxt(path_sequence, delimiter=",")

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        path_label = os.path.join(self.root_dir+ "_label", "machine-" + str(self.subject_id) + ".txt")
        self.label = np.loadtxt(path_label)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class SMDDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).to(torch.float32)
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).to(torch.float32)
            negative = torch.Tensor(negative).to(torch.float32)
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, "machine-" + str(self.subject_id) + ".txt")
        self.sequence = np.loadtxt(path_sequence, delimiter=",")

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        path_label = os.path.join(self.root_dir+ "_label", "machine-" + str(self.subject_id) + ".txt")
        self.label = np.loadtxt(path_label)

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0] - w_size) / stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st + w_size]
            if self.label[st:st + w_size].any() > 0:
                lbl = 1
            else:
                lbl = 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class BoilerDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, (self.subject_id) + ".csv")
        self.sequence = pd.read_csv(path_sequence).values
        self.label = self.sequence[:, -1]
        self.sequence = self.sequence[:, 2:-1].astype(float)

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class BoilerDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, (self.subject_id) + ".csv")
        self.sequence = pd.read_csv(path_sequence).values
        self.label = self.sequence[:, -1]
        self.sequence = self.sequence[:, 2:-1].astype(float)

        # if self.split_type == "test":
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        self.sequence = (self.sequence - self.mean) / self.std

        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

def get_injector(sample_batched, d_mean, d_std):
    sample_batched = (sample_batched * d_std) + d_mean
    injected_window = Injector(sample_batched)
    injected_window.injected_win = (injected_window.injected_win - d_mean) / d_std

    return injected_window.injected_win


def get_output_dim(args):
    output_dim = -1

    if "SMD" in args.path_src:
        output_dim = 1
    elif "MSL" in args.path_src:
        output_dim = 1
    elif "Boiler" in args.path_src:
        output_dim = 1
    else:
        output_dim = 6

    return output_dim

def collate_test(batch):
    #The input is list of dictionaries
    out = {}
    for key in batch[0].keys():
        val = []
        for sample in batch:
            val.append(sample[key])
        val = torch.cat(val, dim=0)
        out[key] = val
    return out



