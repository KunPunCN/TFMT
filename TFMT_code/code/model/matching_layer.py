import torch
from torch import nn
from torch.nn import functional as F


class MatchingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size * 3, 4)

    def gene_pred(self, batch_size, S_preds, E_preds, pairs_true):
        all_pred = [[] for i in range(batch_size)]
        pred_prob = [[] for i in range(batch_size)]
        pred_label = [[] for i in range(batch_size)]
        pred_maxlen = 0
        for i in range(batch_size):
            S_pred = torch.nonzero(S_preds[i]).cpu().numpy()
            E_pred = torch.nonzero(E_preds[i]).cpu().numpy()

            for (s0, s1) in S_pred:
                for (e0, e1) in E_pred:
                    if s0 <= e0 and s1 <= e1:
                        sentiment = 0
                        for j in range(len(pairs_true[i])):
                            p = pairs_true[i][j]
                            if [s0 - 1, e0, s1 - 1, e1] == p[:4]:
                                sentiment = p[4]
                        pred_label[i].append(sentiment)
                        all_pred[i].append([s0 - 1, e0, s1 - 1, e1])
                        # NOTE 1 用于存储标签的概率值
                        pred_prob[i].append(-1)


            if len(all_pred[i]) > pred_maxlen:
                pred_maxlen = len(all_pred[i])

        for i in range(batch_size):
            for j in range(len(all_pred[i]), pred_maxlen):
                pred_label[i].append(-1)
                pred_prob[i].append(-1)
        pred_label = torch.tensor(pred_label).to('cuda')
        pred_prob = torch.tensor(pred_prob).to('cuda')
        return all_pred, pred_label, pred_maxlen, pred_prob

    def input_encoding(self, batch_size, pairs, maxlen, table, seq):
        input_ret = torch.zeros([batch_size, maxlen, self.config.hidden_size * 3]).to('cuda')
        input_ret_S = torch.zeros([batch_size, maxlen, self.config.hidden_size]).to('cuda')
        input_ret_E = torch.zeros([batch_size, maxlen, self.config.hidden_size]).to('cuda')
        for i in range(batch_size):
            j = 0
            for (s0, e0, s1, e1) in pairs[i]:
                S = table[i, s0 + 1, s1 + 1, :]
                E = table[i, e0, e1, :]
                R = torch.max(torch.max(table[i, s0 + 1:e0 + 1, s1 + 1:e1 + 1, :], dim=1)[0], dim=0)[0]
                input_ret[i, j, :] = torch.cat([S, E, R])
                input_ret_S[i, j, :] = S
                input_ret_E[i, j, :] = E
                j += 1
        return input_ret, input_ret_S, input_ret_E


    def forward(self, outputs, Table, pairs_true, seq, mode):
        seq = seq.clone().detach()
        table = Table.clone()
        batch_size = table.size(0)
        # NOTE 计算table_loss
        all_pred, pred_label, pred_maxlen, pred_prob = self.gene_pred(batch_size, outputs['table_predict_S'],
                                                                      outputs['table_predict_E'], pairs_true)

        pred_input, S, E = self.input_encoding(batch_size, all_pred, pred_maxlen, table, seq)
        pred_output = self.linear(pred_input)

        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        loss_input = pred_output
        loss_label = pred_label

        ##
        outputs['pair_loss'] = 0
        outputs['all_preds'] = []
        if mode == 'no':

            if loss_input.shape[1] == 0:
                loss_input = torch.zeros([batch_size, 1, 2])
                loss_label = torch.zeros([batch_size, 1]) - 1

            pair_loss = loss_func(loss_input.transpose(1, 2).contiguous(),
                                  loss_label.long())
            outputs['pair_loss'] = pair_loss

        pairs_logits = F.softmax(pred_output, dim=2)

        outputs['pairs_preds'] = []
        outputs['pseudo_preds'] = []
        outputs['pseudo_logits'] = []
        outputs['pseudo_mask'] = []
        outputs['graph'] = []
        outputs['feature'] = torch.tensor(0).to('cuda')

        if pairs_logits.shape[1] == 0:
            return outputs

        pairs_pred = pairs_logits.argmax(dim=2)

        if mode == 'no':
            for i in range(batch_size):
                for j in range(len(all_pred[i])):
                    se = all_pred[i][j]
                    outputs['all_preds'].append((i, se[0], se[1], se[2], se[3], pairs_pred[i][j].item()))
                    if pairs_pred[i][j] >= 1:
                        outputs['pairs_preds'].append((i, se[0], se[1], se[2], se[3], pairs_pred[i][j].item()))

        elif mode == 'teacher':
            pseudo_all_preds = [[] for i in range(batch_size)]
            pseudo_mask = -torch.ones(pairs_logits.shape[0], pairs_logits.shape[1]).to('cuda')
            pseudo_logits = torch.zeros_like(pairs_logits).to('cuda')
            max_len = 0
            pseudo_preds = pairs_logits.max(dim=2)[0]

            for i in range(batch_size):
                n = 0
                for j in range(len(all_pred[i])):
                    if pairs_pred[i][j] >= 1:
                        se = all_pred[i][j]
                        if pseudo_preds[i][j] >= self.config.pseudo:
                            pseudo_all_preds[i].append([se[0], se[1], se[2], se[3]])
                            pseudo_logits[i, n, :] = pairs_logits[i][j]
                            pseudo_mask[i, n] = 1
                            n += 1


            outputs['pseudo_preds'] = pseudo_all_preds
            outputs['pseudo_logits'] = pseudo_logits
            outputs['pseudo_mask'] = pseudo_mask

        pred_mask = torch.where(pairs_pred>0, True, False).unsqueeze(-1)
        x = torch.masked_select(pred_input, pred_mask).view(-1, self.config.hidden_size*3)
        outputs['feature'] = x

        outputs['feature3'] = torch.masked_select(S, pred_mask).view(-1, self.config.hidden_size)
        outputs['feature4'] = torch.masked_select(E, pred_mask).view(-1, self.config.hidden_size)

        return outputs

    def student_forward(self, outputs, Table, pairs_true, seq, mode, pseudo_preds, pseudo_mask):
        seq = seq.clone().detach()
        table = Table.clone()

        outputs['pseudo_logits'] = []
        outputs['feature'] = torch.tensor(0).to('cuda')
        if pseudo_mask == []:
            return outputs
        batch_size, pred_maxlen = pseudo_mask.size(0), pseudo_mask.size(1)

        pred_input, S, E = self.input_encoding(batch_size, pseudo_preds, pred_maxlen, table, seq)
        pred_output = self.linear(pred_input)

        pairs_logits = F.softmax(pred_output, dim=2)

        if pairs_logits.shape[1] == 0:
            return outputs

        outputs['pseudo_logits'] = pairs_logits

        pred_mask = torch.where(pseudo_mask == -1, False, True).unsqueeze(-1)#.expand_as(pred_input)
        x = torch.masked_select(pred_input, pred_mask).view(-1, self.config.hidden_size * 3)
        outputs['feature'] = x

        outputs['feature3'] = torch.masked_select(S, pred_mask).view(-1, self.config.hidden_size)
        outputs['feature4'] = torch.masked_select(E, pred_mask).view(-1, self.config.hidden_size)
        return outputs

