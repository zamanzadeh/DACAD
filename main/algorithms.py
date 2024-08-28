import sys
sys.path.append("..")
import os
import numpy as np
import math
import torch
import torch.nn as nn
from utils.dataset import get_output_dim
from utils.mlp import MLP
from utils.tcn_no_norm import TemporalConvNet
from utils.augmentations import Augmenter, concat_mask
from utils.util_progress_log import AverageMeter, PredictionMeter, get_dataset_type
from utils.loss import PredictionLoss, SupervisedContrastiveLoss
from models.dacad import DACAD_NN
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


#Given the args, it will return the algorithm directly
def get_algorithm(args, input_channels_dim, input_static_dim):
    if args.algo_name == "dacad":
        return DACAD(args, input_channels_dim, input_static_dim)
    else:
        return None

def get_num_channels(args):
    return list(map(int, args.num_channels_TCN.split("-")))

class Base_Algorithm(nn.Module):
    def __init__(self, args):
        super(Base_Algorithm, self).__init__()

        #Record the args if needed later on
        self.args = args

        #Let algorithm know its name and dataset_type
        self.algo_name = args.algo_name
        self.dataset_type = get_dataset_type(args)

        self.pred_loss = PredictionLoss(self.dataset_type, args.task, args.weight_ratio)
        self.sup_cont_loss = SupervisedContrastiveLoss(self.dataset_type)

        self.output_dim = get_output_dim(args)

        #Only used for TCN-related models
        self.num_channels = get_num_channels(args)

        #During training we report one main metric
        self.main_pred_metric = ""
        if self.dataset_type == "smd":
            self.main_pred_metric = "avg_prc"
        elif self.dataset_type == "msl":
            self.main_pred_metric = "avg_prc"
        elif self.dataset_type == "boiler":
            self.main_pred_metric = "avg_prc"
        #If it is sensor data, we will use Macro f1
        else:
            self.main_pred_metric = "mac_f1"


        self.init_pred_meters_val()

    def init_pred_meters_val(self):
            #We save prediction scores for validation set (for reporting purposes)
            self.pred_meter_val_src = PredictionMeter(self.args)
            self.pred_meter_val_trg = PredictionMeter(self.args)

    def predict_trg(self, sample_batched):
        trg_feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        y_pred_trg = self.classifier(trg_feat, sample_batched.get('static'))

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):
        src_feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        y_pred_src = self.classifier(src_feat, sample_batched.get('static'))

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))
    
    def get_embedding(self, sample_batched):
        feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        return feat

    #Score prediction is dataset and task dependent, that's why we write init function here
    def init_score_pred(self):
        if self.dataset_type == "smd":
            return AverageMeter('ROC AUC', ':6.2f')
        elif self.dataset_type == "msl":
            return AverageMeter('ROC AUC', ':6.2f')
        elif self.dataset_type == "boiler":
            return AverageMeter('ROC AUC', ':6.2f')
        else:
            return AverageMeter('Macro F1', ':6.2f')

    #Helper function to build TCN feature extractor for all related algorithms 
    def build_feature_extractor_TCN(self, args, input_channels_dim, num_channels):
        return TemporalConvNet(num_inputs=input_channels_dim, num_channels=num_channels, kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN, dropout=args.dropout)



#DACAD Algorithm
class DACAD(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(DACAD, self).__init__(args)

        self.input_channels_dim = input_channels_dim
        self.input_static_dim = input_static_dim

        #different from other algorithms, we import entire model at onces. (i.e. no separate feature extractor or classifier)
        self.model = DACAD_NN(num_inputs=(1+args.use_mask)*input_channels_dim, output_dim=self.output_dim, num_channels=self.num_channels, num_static=input_static_dim,
            mlp_hidden_dim=args.hidden_dim_MLP, use_batch_norm=args.use_batch_norm, kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN, dropout=args.dropout, K=args.queue_size, m=args.momentum)

        self.augmenter = None
        self.concat_mask = concat_mask

        self.criterion_CL = nn.CrossEntropyLoss()

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()

    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        p = float(kwargs.get("count_step")) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        seq_k_src, seq_k_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']
        seq_q_src, seq_q_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']
        # compute output
        output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s,\
        center, squared_radius, q_s_repr, q_s_pos, q_s_neg, p_q_s, p_q_s_pos, p_q_s_neg, q_t, q_t_pos, q_t_neg, p_q_t, p_q_t_pos, p_q_t_neg = \
            self.model(sample_batched_src['sequence'], seq_k_src, sample_batched_src['label'], sample_batched_src.get('static'),
                       sample_batched_trg['sequence'], seq_k_trg, sample_batched_trg.get('static'), alpha,
                       sample_batched_src['positive'], sample_batched_src['negative'], sample_batched_trg['positive'], sample_batched_trg['negative'])

        # Compute all losses
        loss_disc = F.binary_cross_entropy_with_logits(output_disc, target_disc)
        # criterion = nn.BCELoss()
        # loss_disc = criterion(output_disc, target_disc)

        # Task classification  Loss
        src_cls_loss = self.pred_loss.deep_svdd_loss(q_s_repr, sample_batched_src['label'], center, squared_radius)

        src_sup_cont_loss = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_s, p_q_s_pos, p_q_s_neg, sample_batched_src['label'], margin=2)
        trg_fake_labels = np.zeros(len(sample_batched_trg['label']))
        trg_inj_cont_loss = self.sup_cont_loss.get_sup_cont_tripletloss(p_q_t, p_q_t_pos, p_q_t_neg, trg_fake_labels, margin=2)

        tmp_loss = 0
        if (trg_inj_cont_loss - src_sup_cont_loss) < 0 :
            tmp_loss = abs(trg_inj_cont_loss - src_sup_cont_loss)


        loss = self.args.weight_loss_disc*loss_disc + self.args.weight_loss_pred*src_cls_loss \
               + self.args.weight_loss_src_sup * src_sup_cont_loss + self.args.weight_loss_trg_inj * trg_inj_cont_loss + tmp_loss

        #If in training mode, do the backprop
        if self.training:
            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.losses_sup.update(src_sup_cont_loss.item(), seq_q_src.size(0))
        self.losses_inj.update(trg_inj_cont_loss.item(), seq_q_trg.size(0))

        acc1 = accuracy_score(output_disc.detach().cpu().numpy().flatten()>0.5, target_disc.detach().cpu().numpy().flatten())
        self.losses_disc.update(loss_disc.item(), output_disc.size(0))
        self.top1_disc.update(acc1, output_disc.size(0))

        self.losses_pred.update(src_cls_loss.item(), seq_q_src.size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], pred_s)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], pred_s)

            #keep track of prediction results (of target) explicitly
            pred_t = self.model.predict(sample_batched_trg, sample_batched_trg.get('static'), is_target=True, is_eval=False)

            self.pred_meter_val_trg.update(sample_batched_trg['label'], pred_t)

    def init_metrics(self):
        self.losses_s = AverageMeter('Loss Source', ':.4e')
        self.top1_s = AverageMeter('Acc@1', ':6.2f')
        self.losses_sup = AverageMeter('L_Src_Sup', ':.4e')
        self.top1_sup = AverageMeter('Acc@1', ':6.2f')
        self.losses_inj = AverageMeter('L_Trg_Inj', ':.4e')
        self.top1_inj = AverageMeter('Acc@1', ':6.2f')
        self.losses_t = AverageMeter('Loss Target', ':.4e')
        self.top1_t = AverageMeter('Acc@1', ':6.2f')
        self.losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.losses_disc = AverageMeter('Loss Disc', ':.4e')
        self.top1_disc = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_sup, self.losses_inj, self.losses_disc, self.losses_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    #We need to overwrite below functions for DACAD
    def predict_trg(self, sample_batched):

        seq_t = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_trg = self.model.predict(sample_batched, sample_batched.get('static'), is_target=True, is_eval=True)

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):

        seq_s = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_src = self.model.predict(sample_batched, sample_batched.get('static'), is_target=False, is_eval=True)

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def get_embedding(self, sample_batched):

        seq = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        feat = self.model.get_encoding(seq)

        return feat

    def get_augmenter(self, sample_batched):

        seq_len = sample_batched['sequence'].shape[1]
        num_channel = sample_batched['sequence'].shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        else:
            print("The model only support multi channel time series data")


