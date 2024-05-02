import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from .table import TableEncoder
from .matching_layer import MatchingLayer


class BDTFModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.table_encoder = TableEncoder(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)

        self.init_weights()

    def forward(self, input_ids, input_ids2, attention_mask, ids,
                table_labels_S=None, table_labels_E=None, domain_ids=None,
                polarity_labels=None, pairs_true=None, domain_id=None, epoch=None, mode='no', pseudo_preds=None,
                pseudo_mask=None):


        if mode == 'teacher':
            input_ids = input_ids2

        seq = self.bert(input_ids, attention_mask)[0]

        table,_ = self.table_encoder(seq, attention_mask)

        output = self.inference(table, attention_mask, table_labels_S, table_labels_E, domain_ids, mode)

        output['ids'] = ids

        if mode == 'student':
            output = self.matching.student_forward(output, table, pairs_true, seq, mode, pseudo_preds, pseudo_mask)
        else:
            output = self.matching(output, table, pairs_true, seq, mode)

        return output


class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 2
        length = ((attention_mask.sum(dim=1) - 2) * z).long()
        length[length < 5] = 5
        max_length = mask_length ** 2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True)
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length - 1].unsqueeze(1)
        return pred >= (topkth.view(batch_size, 1, 1))

    def forward(self, table, attention_mask, table_labels_S, table_labels_E, domain_ids, mode):
        outputs = {}

        logits_S = torch.squeeze(self.cls_linear_S(table), 3)
        logits_E = torch.squeeze(self.cls_linear_E(table), 3)

        outputs['table_loss_S'] = 0
        outputs['table_loss_E'] = 0
        if mode == 'no':
            mask = table_labels_S
            loss_func = nn.BCEWithLogitsLoss(weight=(mask >= 0))
            outputs['table_loss_S'] = loss_func(logits_S, table_labels_S.float())
            outputs['table_loss_E'] = loss_func(logits_E, table_labels_E.float())

        S_pred = torch.sigmoid(logits_S) * (table_labels_S >= 0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S >= 0)

        if self.config.span_pruning != 0:
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask)
        else:
            table_predict_S = S_pred > 0.5
            table_predict_E = E_pred > 0.5

        outputs['logits_S'] = logits_S
        outputs['logits_E'] = logits_E
        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E

        return outputs


