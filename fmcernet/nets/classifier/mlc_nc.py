import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .meta_classifier import MetaClassifier
from fmcernet.utils import ExtendMultiLabelMetric, build_evaluator


nINF = -100


def build_etf_matrix(feat_dim, num_classes):
    if num_classes < 2:
        raise ValueError(f"ETF requires at least 2 classes, got {num_classes}.")
    if feat_dim < num_classes:
        raise ValueError(
            f"ETF requires feat_dim >= num_classes, got {feat_dim} < {num_classes}."
        )
    random_matrix = torch.rand(feat_dim, num_classes)
    orthogonal_basis, _ = torch.linalg.qr(random_matrix, mode='reduced')
    eye = torch.eye(num_classes)
    one = torch.ones(num_classes, num_classes)
    simplex = eye - one / num_classes
    return math.sqrt(num_classes / (num_classes - 1)) * orthogonal_basis @ simplex


class TwoWayLoss(nn.Module):
    def __init__(self, tp=4.0, tn=1.0):
        super().__init__()
        self.tp = tp
        self.tn = tn

    def forward(self, logits, labels):
        class_mask = (labels > 0).any(dim=0)
        sample_mask = (labels > 0).any(dim=1)

        pmask = labels.masked_fill(labels <= 0, nINF).masked_fill(labels > 0, 0.0)
        nmask = labels.masked_fill(labels != 0, nINF).masked_fill(labels == 0, 0.0)

        plogit_class = torch.logsumexp(-logits / self.tp + pmask, dim=0).mul(self.tp)[class_mask]
        nlogit_class = torch.logsumexp(logits / self.tn + nmask, dim=0).mul(self.tn)[class_mask]
        plogit_sample = torch.logsumexp(-logits / self.tp + pmask, dim=1).mul(self.tp)[sample_mask]
        nlogit_sample = torch.logsumexp(logits / self.tn + nmask, dim=1).mul(self.tn)[sample_mask]

        if not class_mask.any() or not sample_mask.any():
            return logits.sum() * 0.0

        return (
            F.softplus(nlogit_class + plogit_class).mean()
            + F.softplus(nlogit_sample + plogit_sample).mean()
        )


class FeatureLabelLoss(nn.Module):
    def forward(self, features, embeddings, labels):
        features_norm = F.normalize(features, p=2, dim=2)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        embeddings_norm = embeddings_norm.unsqueeze(0).expand(features.size(0), -1, -1)
        cosine = torch.bmm(features_norm, embeddings_norm.transpose(1, 2))
        similarities = torch.diagonal(cosine, dim1=-1, dim2=-2)

        num_classes = embeddings.size(0)
        pos_score = (1 + similarities) / 2
        neg_score = 1 - (num_classes - 1) / num_classes * torch.abs(
            1 / (num_classes - 1) + similarities
        )

        pos_loss = labels * torch.log(pos_score.clamp(min=1e-6))
        neg_loss = (1 - labels) * torch.log(neg_score.clamp(min=1e-6))
        return -(pos_loss + neg_loss).mean()


class ContrastiveProtoLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, class_prototypes, feature_proj, labels, prototype_ready):
        positive_mask = labels.bool() & prototype_ready.unsqueeze(0)
        if not positive_mask.any():
            return feature_proj.sum() * 0.0

        feature_proj = F.normalize(feature_proj, dim=2)
        class_prototypes = F.normalize(class_prototypes, dim=1)
        positive_features = feature_proj[positive_mask]
        positive_labels = positive_mask.nonzero(as_tuple=False)[:, 1]
        logits = positive_features @ class_prototypes.T / self.temperature
        return F.cross_entropy(logits, positive_labels)


class MLCNC(MetaClassifier):
    def __init__(self, args):
        evaluator = build_evaluator([ExtendMultiLabelMetric(
            thr=args.positive_thr,
            num_classes=args.num_classes,
            logger_name=args.logger_name,
            with_binary=False,
        )])
        super().__init__(evaluator, args)

        cfg = args.get('mlc_nc_cfg', {})
        self.num_classes = args.num_classes
        self.with_background = cfg.get('with_background', True)
        self.num_output_classes = self.num_classes + int(self.with_background)
        self.background_index = self.num_output_classes - 1
        self.input_embed_dim = args.backbone_cfg['backbone_token_output_dim'][-1]
        self.embed_dim = cfg.get('embed_dim', 768)
        self.project_dim = cfg.get('project_dim', 20)
        self.alpha = cfg.get('alpha', 1.0)
        self.classifier_type = cfg.get('classifier', 'ETF')
        self.prototype_momentum = cfg.get('prototype_momentum', 0.9)

        if self.classifier_type not in ['ETF', 'GroupFC']:
            raise ValueError(f"Unsupported MLC-NC classifier: {self.classifier_type}")

        self.embed_standart = nn.Linear(self.input_embed_dim, self.embed_dim)
        self.query_embed_group = nn.Embedding(self.num_output_classes, self.embed_dim)
        self.register_buffer(
            'query_embed_etf',
            build_etf_matrix(self.embed_dim, self.num_output_classes).T,
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=cfg.get('num_heads', 8),
            dropout=cfg.get('dropout', 0.1),
        )

        self.duplicate_pooling = nn.Parameter(torch.empty(self.num_output_classes, self.embed_dim, 1))
        self.duplicate_pooling_bias = nn.Parameter(torch.empty(self.num_output_classes))
        self.duplicate_proto = nn.Parameter(torch.empty(self.num_output_classes, self.embed_dim, self.project_dim))
        self.duplicate_proto_bias = nn.Parameter(torch.empty(self.num_output_classes, self.project_dim))
        self.proto_classifier = nn.Parameter(
            build_etf_matrix(self.num_output_classes * self.project_dim, self.num_output_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1))

        nn.init.xavier_normal_(self.duplicate_pooling)
        nn.init.constant_(self.duplicate_pooling_bias, 0)
        nn.init.xavier_normal_(self.duplicate_proto)
        nn.init.constant_(self.duplicate_proto_bias, 0)

        self.two_way_loss = TwoWayLoss(
            tp=cfg.get('two_way_tp', 4.0),
            tn=cfg.get('two_way_tn', 1.0),
        )
        self.feature_label_loss = FeatureLabelLoss()
        self.contrastive_proto_loss = ContrastiveProtoLoss(
            temperature=cfg.get('prototype_temperature', 0.5)
        )
        self.weight_twoway = cfg.get('weight_twoway', 1.0)
        self.weight_fla = cfg.get('weight_fla', 0.5)
        self.weight_prototype = cfg.get('weight_prototype', 0.1)

        self.register_buffer(
            'class_prototypes',
            torch.zeros(self.num_output_classes, self.project_dim),
        )
        self.register_buffer('class_prototype_counts', torch.zeros(self.num_output_classes))

    def get_loss_labels(self, databatch):
        labels = self.get_mlc_labels(databatch)
        if not self.with_background:
            return labels

        background = (labels.sum(dim=1, keepdim=True) == 0).to(labels.dtype)
        return torch.cat([labels, background], dim=1)

    def calc_logits(self, inputs):
        img_tokens = self.get_img_tokens(inputs)
        embedding_spatial = self.embed_standart(img_tokens)
        embedding_spatial = F.relu(embedding_spatial, inplace=False)

        if self.classifier_type == 'GroupFC':
            query_embed = self.query_embed_group.weight
        elif self.classifier_type == 'ETF':
            query_embed = self.query_embed_etf
        else:
            raise ValueError(f"Unsupported MLC-NC classifier: {self.classifier_type}")

        batch_size = embedding_spatial.shape[0]
        tgt = query_embed.unsqueeze(1).expand(-1, batch_size, -1)
        h, _ = self.multihead_attn(
            tgt,
            embedding_spatial.transpose(0, 1),
            embedding_spatial.transpose(0, 1),
        )
        h = h.transpose(0, 1)
        h = torch.where(h > 0, h * self.alpha, h)

        feature_proj = torch.einsum('bce,ced->bcd', h, self.duplicate_proto)
        feature_proj = feature_proj + self.duplicate_proto_bias

        if self.classifier_type == 'GroupFC':
            logits = torch.einsum('bce,ceo->bco', h, self.duplicate_pooling).squeeze(-1)
            logits = logits + self.duplicate_pooling_bias
        elif self.classifier_type == 'ETF':
            feature_proj = F.normalize(feature_proj, dim=-1)
            logits = feature_proj.flatten(1) @ self.proto_classifier
        else:
            raise ValueError(f"Unsupported MLC-NC classifier: {self.classifier_type}")

        return self.temperature * logits, h, query_embed, feature_proj

    def _update_class_prototypes(self, feature_proj, labels):
        with torch.no_grad():
            batch_sums = torch.einsum('bc,bcd->cd', labels, feature_proj.detach())
            batch_counts = labels.sum(dim=0)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_sums)
                dist.all_reduce(batch_counts)

            present = batch_counts > 0
            if not present.any():
                return

            batch_means = batch_sums[present] / batch_counts[present].unsqueeze(1)
            seen = self.class_prototype_counts[present] > 0
            updated = self.class_prototypes[present].clone()
            updated[~seen] = batch_means[~seen]
            updated[seen] = (
                self.prototype_momentum * updated[seen]
                + (1 - self.prototype_momentum) * batch_means[seen]
            )
            self.class_prototypes[present] = updated
            self.class_prototype_counts[present] += batch_counts[present]

    def calc_loss(self, inputs, databatch):
        logits, features, embeddings, feature_proj = self.calc_logits(inputs)
        labels = self.get_loss_labels(databatch)
        prototype_ready = self.class_prototype_counts > 0

        loss_twoway = self.two_way_loss(logits, labels)
        loss_fla = self.feature_label_loss(features, embeddings, labels)
        loss_proto = self.contrastive_proto_loss(
            self.class_prototypes.detach(),
            feature_proj,
            labels,
            prototype_ready,
        )
        loss = (
            self.weight_twoway * loss_twoway
            + self.weight_fla * loss_fla
            + self.weight_prototype * loss_proto
        )
        if self.training:
            self._update_class_prototypes(feature_proj, labels)

        loss_dict = {
            'twoway_loss': loss_twoway.item(),
            'fla_loss': loss_fla.item(),
            'proto_loss': loss_proto.item(),
        }
        return loss, loss_dict

    def set_pred(self, inputs, databatch):
        logits, features, _, feature_proj = self.calc_logits(inputs)
        if self.with_background:
            pos_logits = logits[:, :self.num_classes]
            bg_logits = logits[:, self.background_index:self.background_index + 1]
            pos_probs = torch.sigmoid(pos_logits - bg_logits)
            feature_proj = feature_proj[:, :self.num_classes]
        else:
            pos_probs = torch.sigmoid(logits)

        pos_prob_feat = torch.cat([pos_probs.unsqueeze(-1), feature_proj], dim=2)

        data_samples = []
        for item, pos_p, img_token in zip(databatch['data_samples'], pos_probs, pos_prob_feat):
            item.pos_prob = pos_p
            item.img_token = img_token
            data_samples.append(item)
        return data_samples
