import os
import torch
import logging
import argparse
import random
import numpy as np

import pytorch_lightning as pl

pl.seed_everything(42)

polarity_map_reversed = {
    1: 'NEG',
    2: 'NEU',
    3: 'POS'
}

from transformers import AutoTokenizer, AutoConfig
from transformers.optimization import AdamW
from pytorch_lightning.utilities import rank_zero_info
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_constant_schedule,#NOTE 因为是继续训练不需要warmup
)

arg_to_scheduler = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_w_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule_with_warmup,
}

from model.bdtf_model import BDTFModel#bdtf_model
from utils.aste_datamodule import ASTEDataModule#aste_datamodule
from utils.aste_result import Result
from utils import params_count

logger = logging.getLogger(__name__)

from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


class ASTE(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
        self.config.table_num_labels = self.data_module.table_num_labels
        self.config.table_encoder = self.hparams.table_encoder
        self.config.num_table_layers = self.hparams.num_table_layers
        self.config.span_pruning = self.hparams.span_pruning
        self.config.seq2mat = self.hparams.seq2mat
        self.config.num_d = self.hparams.num_d
        self.patience = 3
        self.current_val_n = 0

        self.config.alpha = 0.7
        self.config.span_pruning = 0.3
        self.config.pseudo = 0.98

        self.config.region_l = 1
        self.config.mmd_l = 5e-3
        self.config.temperature = 1

        self.model = BDTFModel.from_pretrained(self.hparams.model_name_or_path, config=self.config)
        self.teacher = BDTFModel.from_pretrained(self.hparams.model_name_or_path, config=self.config)


        print('---------------------------------------')
        print('total params_count:', params_count(self.model))
        print('---------------------------------------')

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir, str(self.hparams.cuda_ids), 'model')
        print(f'## save model to {dir_name}')
        print('\n')
        self.model.save_pretrained(dir_name)

    def load_model(self, dir=None):
        if dir == None:
            dir_name = os.path.join(self.hparams.output_dir, str(self.hparams.cuda_ids), 'model')
        else:
            dir_name = os.path.join(self.hparams.output_dir, dir, 'model')
        print(f'## load model from {dir_name}')
        self.model = BDTFModel.from_pretrained(dir_name)
        self.teacher = BDTFModel.from_pretrained(dir_name)

    def forward(self, **inputs):
        outputs = self.model(**inputs, epoch=self.current_epoch)
        return outputs

    def forward2(self, model, **inputs):
        outputs = model(**inputs, epoch=self.current_epoch, mode='teacher')
        return outputs

    def forward3(self, model, pseudo_preds, pseudo_mask, **inputs):
        outputs = model(**inputs, epoch=self.current_epoch, mode='student', pseudo_preds=pseudo_preds, pseudo_mask=pseudo_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        # batch_size = len(batch)
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        source, target = batch[0]['source'], batch[0]['target']

        loss = torch.tensor(0).to('cuda')
        if source:
            outputs = self.forward(**source)
            loss = loss + outputs['table_loss_S'] + outputs['table_loss_E'] + outputs['pair_loss']
        if target:
            teacher_output = self.forward2(self.teacher, **target)
            pseudo_preds = teacher_output['pseudo_preds']
            pseudo_mask = teacher_output['pseudo_mask']
            teacher_logits = teacher_output['pseudo_logits']

            if teacher_logits == []:
                return loss
            student_output = self.forward3(self.model, pseudo_preds, pseudo_mask, **target)
            student_logits = student_output['pseudo_logits']


            pseudo_mask = torch.clamp(pseudo_mask, min=0)
            region_loss = 0
            if torch.sum(pseudo_mask) != 0:
                region_loss = torch.mean((teacher_logits - student_logits) ** 2, dim=-1)
                region_loss = torch.sum(region_loss * pseudo_mask) / torch.sum(pseudo_mask)

            loss = loss + region_loss * self.config.region_l
            # loss.requires_grad_(True)

        if source and target:
            sour = outputs['feature']
            targ = student_output['feature']
            if targ.shape[0]>0:
                mmd_loss = self.mmd(sour, targ)
                mmd_loss = mmd_loss + self.mmd(outputs['feature3'],student_output['feature3']) + self.mmd(outputs['feature4'], student_output['feature4'])
                loss = loss + mmd_loss * self.config.mmd_l

        self.log('train_loss', loss)

        if loss == torch.tensor(0):
            return None
        return loss


    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=4, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=4, fix_sigma=None):
        if len(source.shape) <= 1:
            source = source.unsqueeze(0)
        if len(target.shape) <= 1:
            target = target.unsqueeze(0)
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                       fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss


    def training_epoch_end(self, training_step_outputs):

        with torch.no_grad():
            for (t_param,s_param) in zip(self.teacher.parameters(), self.model.parameters()):
                t_param.data = self.config.alpha * t_param.data + (1-self.config.alpha) * s_param.data


    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs['table_loss_S'] + outputs['table_loss_E'] + outputs['pair_loss']
        self.log('valid_loss', loss)

        return {
            'ids': outputs['ids'],
            'pair_preds': outputs['pairs_preds'],
            'all_preds': outputs['all_preds'],
        }

    def validation_epoch_end(self, outputs):
        print('Epoch: ', self.current_epoch)
        examples = self.data_module.raw_datasets['dev']

        self.current_val_result = Result.parse_from(outputs, examples)
        self.current_val_result.cal_metric()


        if not hasattr(self, 'best_val_result'):
            self.best_val_result = self.current_val_result

        elif self.best_val_result < self.current_val_result:
            self.best_val_result = self.current_val_result
            print(f'\n')
            self.save_model()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        examples = self.data_module.raw_datasets['test']

        self.test_result = Result.parse_from(outputs, examples)
        self.test_result.cal_metric()


    def save_test_result(self):
        dir_name = os.path.join(self.hparams.output_dir, 'result')
        self.test_result.save(dir_name, self.hparams)

    def setup(self, stage):
        if stage == 'fit':

            self.train_loader = self.train_dataloader()
            dataset_size = max(len(self.data_module.raw_datasets['train']), len(self.data_module.raw_datasets['target']))

            ngpus = (len(self.hparams.gpus.split(',')) if type(self.hparams.gpus) is str else self.hparams.gpus)
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * ngpus
            self.total_steps = (dataset_size / effective_batch_size) * self.hparams.max_epochs


    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps * 2)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return scheduler

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        # freeze_dict = ['inference']

        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)

        #NOTE 冻结inference参数
        for n, p in self.teacher.named_parameters():
            p.requires_grad = False

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if (not has_keywords(n, no_decay) and p.requires_grad == True)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if (has_keywords(n, no_decay) and p.requires_grad == True)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)
        parser.add_argument("--lr_scheduler", type=str)

        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        parser.add_argument("--table_encoder", type=str, default='resnet', choices=['resnet', 'none'])
        parser.add_argument("--num_table_layers", type=int, default=2)
        parser.add_argument("--span_pruning", type=float, default=0.3)
        parser.add_argument("--seq2mat", type=str, default='none',
                            choices=['none', 'tensor', 'context', 'tensorcontext'])
        parser.add_argument("--num_d", type=int, default=64)
        replace_sampler_ddp = False
        return parser


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        # if dist.get_rank() == 0:
        pl_module.current_val_result.report()

    def on_test_end(self, trainer, pl_module):
        # if dist.get_rank() == 0:
        pl_module.test_result.report()
        # pl_module.save_test_result()



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ASTE.add_model_specific_args(parser)
    parser = ASTEDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    data_module = ASTEDataModule.from_argparse_args(args)
    data_module.load_dataset()
    model = ASTE(args, data_module)

    logging_callback = LoggingCallback()

    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        # 'num_sanity_val_steps': 5 if args.do_train else 0,
        'num_sanity_val_steps': 0, # NOTE validation sanity check
        'reload_dataloaders_every_epoch' : True,
        # 'accelerator' : "ddp",
        # 'num_processes' : 1,
        # 'progress_bar_refresh_rate' : 0,
    }

    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    # model.load_model('ppp')

    trainer.fit(model, datamodule=data_module)
    model.load_model()
    trainer.test(model, datamodule=data_module)

def test():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ASTE.add_model_specific_args(parser)
    parser = ASTEDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    data_module = ASTEDataModule.from_argparse_args(args)
    data_module.load_dataset()
    model = ASTE(args, data_module)

    logging_callback = LoggingCallback()

    kwargs = {
        'weights_summary': None,
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        # 'num_sanity_val_steps': 5 if args.do_train else 0,
        'num_sanity_val_steps': 0,
        'reload_dataloaders_every_epoch' : True,
        # 'accelerator': "ddp",
    }

    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    # model.load_model('5')
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()
    #test()

