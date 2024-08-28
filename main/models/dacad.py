import sys

sys.path.append("../..")
import torch
import torch.nn as nn
# Import our encoder
from utils.tcn_no_norm import TemporalConvNet
from utils.mlp import MLP
from main.models.models import DeepSVDD

# Helper function for reversing the discriminator backprop
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DACAD_NN(nn.Module):

    def __init__(self, num_inputs, output_dim, num_channels, num_static, mlp_hidden_dim=256,
                 use_batch_norm=True, num_neighbors=1, kernel_size=2, stride=1, dilation_factor=2,
                 dropout=0.2, K=24576, m=0.999, T=0.07):

        super(DACAD_NN, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.K = K
        self.m = m
        self.T = T
        self.num_neighbors = num_neighbors

        # encoders
        # num_classes is the output fc dimension
        self.encoder_q = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size,
                                         stride=stride, dilation_factor=dilation_factor, dropout=dropout)
        self.encoder_k = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size,
                                         stride=stride, dilation_factor=dilation_factor, dropout=dropout)

        # projector for query
        self.projector = MLP(input_dim=num_channels[-1], hidden_dim=mlp_hidden_dim,
                             output_dim=num_channels[-1], use_batch_norm=use_batch_norm)

        # Classifier trained by source query
        # self.predictor0 = MLP(input_dim = num_channels[-1] + num_static, hidden_dim = mlp_hidden_dim,
        #                        output_dim = output_dim, use_batch_norm = use_batch_norm)

        self.predictor = DeepSVDD(input_dim=num_channels[-1] + num_static,
                                  hidden_dim=mlp_hidden_dim,
                                  output_dim=num_channels[-1] + num_static,
                                  use_batch_norm=use_batch_norm)

        # Discriminator
        # self.discriminator = MLP(input_dim = num_channels[-1], hidden_dim = mlp_hidden_dim,
        #                        output_dim = 1, use_batch_norm = use_batch_norm)

        self.discriminator = Discriminator(input_dim=num_channels[-1], hidden_dim=mlp_hidden_dim,
                                           output_dim=1)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_s", torch.randn(num_channels[-1], K))
        self.queue_s = nn.functional.normalize(self.queue_s, dim=0)

        self.register_buffer("queue_t", torch.randn(num_channels[-1], K))
        self.queue_t = nn.functional.normalize(self.queue_t, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # No update during evaluation
        if self.training:
            # Update the encoder
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        # No update during evaluation
        if self.training:
            # gather keys before updating queue
            batch_size = keys_s.shape[0]

            ptr = int(self.queue_ptr)
            # For now, ignore below assertion
            # assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue_s[:, ptr:ptr + batch_size] = keys_s.T
            self.queue_t[:, ptr:ptr + batch_size] = keys_t.T

            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr

    def forward(self, sequence_q_s, sequence_k_s, real_s, static_s, sequence_q_t, sequence_k_t, static_t, alpha,
                seq_src_positive, seq_src_negative, seq_trg_positive, seq_trg_negative):

        # compute query features
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
        q_s = self.encoder_q(sequence_q_s.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s = nn.functional.normalize(q_s, dim=1)

        q_s_pos = self.encoder_q(seq_src_positive.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s_pos = nn.functional.normalize(q_s_pos, dim=1)

        q_s_neg = self.encoder_q(seq_src_negative.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s_neg = nn.functional.normalize(q_s_neg, dim=1)

        # Project the query
        p_q_s = self.projector(q_s, None)  # queries: NxC
        p_q_s = nn.functional.normalize(p_q_s, dim=1)

        p_q_s_pos = self.projector(q_s_pos, None)  # queries: NxC
        p_q_s_pos = nn.functional.normalize(p_q_s_pos, dim=1)

        p_q_s_neg = self.projector(q_s_neg, None)  # queries: NxC
        p_q_s_neg = nn.functional.normalize(p_q_s_neg, dim=1)
        # TARGET DATASET query computations

        # compute query features
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
        q_t = self.encoder_q(sequence_q_t.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_t = nn.functional.normalize(q_t, dim=1)

        q_t_pos = self.encoder_q(seq_trg_positive.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_t_pos = nn.functional.normalize(q_t_pos, dim=1)

        q_t_neg = self.encoder_q(seq_trg_negative.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_t_neg = nn.functional.normalize(q_t_neg, dim=1)

        # Project the query
        p_q_t = self.projector(q_t, None)  # queries: NxC
        p_q_t = nn.functional.normalize(p_q_t, dim=1)

        p_q_t_pos = self.projector(q_t_pos, None)  # queries: NxC
        p_q_t_pos = nn.functional.normalize(p_q_t_pos, dim=1)

        p_q_t_neg = self.projector(q_t_neg, None)  # queries: NxC
        p_q_t_neg = nn.functional.normalize(p_q_t_neg, dim=1)

        l_queue_s = torch.mm(p_q_s, self.queue_s.clone().detach())
        labels_s = torch.arange(p_q_s.shape[0], dtype=torch.long).to(device=p_q_s.device)
        l_queue_t = torch.mm(p_q_t, self.queue_t.clone().detach())
        labels_t = torch.arange(p_q_t.shape[0], dtype=torch.long).to(device=p_q_t.device)

        # DOMAIN DISCRIMINATION Loss
        real_s = real_s.squeeze(1)
        q_n_s = q_s[real_s == 0]
        q_s_reversed = ReverseLayerF.apply(q_s, alpha)

        domain_label_s = torch.ones((len(q_s), 1)).to(device=q_s.device)
        domain_label_t = torch.zeros((len(q_t), 1)).to(device=q_t.device)

        labels_domain = torch.cat([domain_label_s, domain_label_t], dim=0)

        # q_s_reversed = ReverseLayerF.apply(q_s, alpha)
        q_t_reversed = ReverseLayerF.apply(q_t, alpha)

        q_reversed = torch.cat([q_s_reversed, q_t_reversed], dim=0)
        pred_domain = self.discriminator(q_reversed)

        # SOURCE Prediction task
        # y_s = self.predictor(q_s, static_s)
        pred_s, center, squared_radius = self.predictor(q_s, static_s)

        # dequeue and enqueue
        # self._dequeue_and_enqueue(p_k_s, p_k_t)

        logits_s, logits_t, logits_ts, labels_ts = 0, 0, 0, 0

        return logits_s, labels_s, logits_t, labels_t, logits_ts, labels_ts, pred_domain, labels_domain, pred_s, center, squared_radius, \
            q_s, q_s_pos, q_s_neg, p_q_s, p_q_s_pos, p_q_s_neg, q_t, q_t_pos, q_t_neg, p_q_t, p_q_t_pos, p_q_t_neg

    def get_encoding(self, sequence, is_target=True):
        # compute the encoding of a sequence (i.e. before projection layer)
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output

        # We will use the encoder from a given domain (either source or target)

        q = self.encoder_q(sequence.transpose(1, 2))[:, :, -1]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        return q

    def predict(self, sequence, static, is_target=True, is_eval=False):
        # Get the encoding of a sequence from a given domain
        q = self.get_encoding(sequence['sequence'], is_target=is_target)

        # Make the prediction based on the encoding
        # y = self.predictor(q, static)
        dist, center, squared_radius = self.predictor(q, static)

        return dist  # y
