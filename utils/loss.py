import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLoss(object):
    def __init__(self, dataset_type="smd", task="decompensation", weight_ratio=None):
        self.dataset_type = dataset_type
        self.task = task
        self.weight_ratio = weight_ratio
        self.get_loss_weights()

    #Calculate weights for binary classification, such that average will be around 1,
    # yet minority class will have "weight_ratio" times more weight.
    def get_loss_weights(self):
            prop = 1 - (1/(self.weight_ratio + 1))
            weight_0 = 1 / (2 * prop)
            weight_1 = weight_0 * prop / (1 - prop)
            self.loss_weights = (weight_0, weight_1)

    #Return the appropriate prediction loss for the related task
    def get_prediction_loss(self, output, target):
        if self.dataset_type == "smd" or self.dataset_type == "msl":
            target = target.float()
            batch_loss_weights = target * self.loss_weights[1] + (1 - target) * self.loss_weights[0]
            loss = F.binary_cross_entropy(output, target, weight = batch_loss_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
            #Currently, target has the shape (N,1), we need to flatten
            loss = loss_fn(output, target.squeeze(1))
        return loss


    def deep_svdd_loss(self, inputs, labels, center, squared_radius, rec_loss_weight=0.0, nu=0.0):
        # Reconstruction loss
        # rec_loss = torch.mean((inputs - decoded) ** 2)

        # Regularization term (distance to the center)
        repr = inputs.clone()
        dist_to_center = torch.norm(repr - center, dim=1)

        # Regularization term (distance to the radius)
        normal_loss = torch.sqrt(torch.max(torch.zeros_like(squared_radius), dist_to_center - squared_radius)) #torch.sqrt(dist_to_center) * mask #
        anomal_loss = torch.sqrt(torch.max(torch.zeros_like(squared_radius), 1 + squared_radius - dist_to_center))
        labels_src = labels.squeeze()
        labels_src = labels_src.cuda()
        normal_loss = normal_loss.cuda()
        anomal_loss = anomal_loss.cuda()

        dist_to_radius = torch.where(labels_src == 0, normal_loss , anomal_loss)

        target = labels_src.float()
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        t_loss = loss_fn(dist_to_center, target)
        ent_loss = torch.mean(self.loss_weights[0] * t_loss * (1 - target) + self.loss_weights[1] * t_loss * target)

        total_loss = ent_loss

        return total_loss

class SupervisedContrastiveLoss(object):
    def __init__(self, dataset_type="smd"):
        self.dataset_type = dataset_type

    def get_sup_cont_loss(self, output_src1, output_pos1, output_neg1, labels_src):
        """
        Supervised Contrastive loss
        """

        tau = 1.3
        # Filter the batch on Normal anchors
        output_src = output_src1[labels_src.squeeze() == 0]
        output_pos = output_pos1[labels_src.squeeze() == 0]
        output_neg = output_neg1[labels_src.squeeze() == 0]

        # Dot products
        # dot_product_pos = torch.mm(output_src, output_pos.t())  # A x P
        dot_product_pos = output_src * output_pos
        dot_product_neg = torch.mm(output_src, output_neg.t())  # A x N

        # Exponential terms
        exp_pos = torch.exp(dot_product_pos / tau)
        exp_neg = torch.exp(dot_product_neg / tau)

        nominator = exp_pos.sum(dim=1, keepdim=True)

        # Sum across positive and negative examples
        denominator = nominator + exp_neg.sum(dim=1, keepdim=True)

        # Calculate the main fraction inside the log
        fraction = nominator / denominator

        # Calculate the negative log of the fraction
        neg_log = -torch.log(fraction)

        # Sum over P
        sum_over_p = neg_log.sum(dim=1)

        # Multiply by -1/|A|
        final_loss = sum_over_p.mean()

        return final_loss


    def get_sup_cont_tripletloss(self, output_src1, output_pos1, output_neg1, labels_src, margin=1.0):
        """
        Triplet Contrastive loss
        """
        tau = 1
        # Filter the batch on Normal anchors
        output_src = output_src1[labels_src.squeeze() == 0]
        output_pos = output_pos1[labels_src.squeeze() == 0]
        output_neg = output_neg1[labels_src.squeeze() == 0]

        dist_pos = F.pairwise_distance(output_src, output_pos, 2)
        # dist_neg = F.pairwise_distance(output_src, output_neg, 2)
        anchor_expanded = output_src.unsqueeze(1).expand(-1, output_neg.size(0), -1)
        neg_expanded = output_neg.unsqueeze(0).expand(output_src.size(0), -1, -1)
        dist_neg = torch.norm(anchor_expanded - neg_expanded, dim=2)

        loss = torch.relu(dist_pos**2 - dist_neg**2 + margin)

        # Multiply by -1/|A|
        final_loss = (loss/tau).mean()

        return final_loss
