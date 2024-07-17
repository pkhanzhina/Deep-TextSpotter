import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils.neptune_logger import NeptuneLogger
import Levenshtein

from loss.ctc_loss.ctc_loss import CTCLoss
from models.fcn import FCN
from datasets.mjsynth import MJSynth
import datasets.augmentations as aug

from configs.ctc_loss.train_cfg import cfg as train_cfg
from utils.decode_path import decode_best_path


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.start_epoch, self.global_step = 0, 0
        self.max_epoch = cfg.max_epoch

        self.__get_data()
        self.__get_model()

        self.logger = NeptuneLogger(cfg.logger)

    def __get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.device == 'gpu' else "cpu")
        self.model = FCN(self.cfg.dataset.nrof_classes).to(self.device)
        self.criterion = CTCLoss(self.device, workers=self.cfg.workers)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
        print('number of trainable params:\t', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def __get_data(self):
        train_preprocess = transforms.Compose([
            aug.VerticalResize(h=32),
            aug.RandomHorizontalCrop(scale=(0, 4)),
            aug.RandomHorizontalResize(scale=(0.8, 1.2)),
            aug.HorizontalResize(w=500),
            aug.GaussianNoise(mean=0, std=5),
            aug.ImageWhitening(self.cfg.dataset.mean, self.cfg.dataset.std)
        ])
        test_preprocess = transforms.Compose([
            aug.VerticalResize(h=32),
            aug.HorizontalResize(w=500),
            aug.ImageWhitening(self.cfg.dataset.mean, self.cfg.dataset.std),
        ])
        self.train_dataset = MJSynth(self.cfg.dataset, 'train', transforms=train_preprocess)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
            collate_fn=self.train_dataset.collate_fn)

        self.test_dataset = MJSynth(self.cfg.dataset, 'test', transforms=test_preprocess)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
            collate_fn=self.test_dataset.collate_fn)

    def console_log(self, epoch, names, metrics):
        print(f"[{epoch}/{self.max_epoch}] step {self.global_step}:\t",
              ',\t'.join(['{!s}: {:.4f}'.format(name, metric) for name, metric in zip(names, metrics)]))

    def _dump_model(self, epoch):
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }
        if not os.path.exists(self.cfg.checkpoints_dir):
            os.makedirs(self.cfg.checkpoints_dir)
        path_to_save = os.path.join(self.cfg.checkpoints_dir, f'epoch-{epoch}.pt')
        torch.save(state_dict, path_to_save)

    def _load_model(self, epoch):
        path = os.path.join(self.cfg.pretrained_dir, f"epoch-{epoch}.pt")
        start_epoch = 0
        try:
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.global_step = state_dict['global_step']
            start_epoch = state_dict['epoch']
        except Exception as e:
            print(e)
        return start_epoch

    def make_training_step(self, data, update=True):
        inputs, labels, inputs_lengths = data[:-1]
        inputs = inputs.to(self.device)
        logits = self.model(inputs)
        outputs = F.softmax(logits, dim=1)
        inputs_lengths = tuple([np.floor(l / 4).astype(np.int32) for l in inputs_lengths])
        params_list = outputs.detach().cpu().numpy()
        loss = self.criterion(params_list, labels, inputs_lengths)

        if update:
            grad = self.criterion.backward()
            logits.backward(grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss, outputs.detach().cpu().numpy()

    def fit(self):
        if self.cfg.epoch_to_load is not None:
            self.start_epoch = self._load_model(self.cfg.epoch_to_load)

        if self.cfg.evaluate_before_training:
            self.evaluate('test', step=self.start_epoch)

        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for batch in pbar:
                loss, output = self.make_training_step(batch)

                gt_words = batch[-1]
                pred_words = self.train_dataset.seq_to_word(decode_best_path(output))

                acc = np.sum([
                    1 - (Levenshtein.distance(pred, gt) / max(len(pred), len(gt)))
                    for pred, gt in zip(pred_words, gt_words)
                ]) / len(gt_words)

                pbar.set_description('[{}] loss - {:.4f}, acc - {:.4f}'.format(epoch, loss, acc))

                self.logger.log_metrics(['train/loss', 'train/acc', 'lr'],
                                        [loss, acc, self.optimizer.param_groups[0]['lr']],
                                        self.global_step)

                self.global_step += 1
                self.scheduler.step()

            # self._dump_model(epoch + 1)
            self.evaluate('test', step=epoch + 1)

    @torch.no_grad()
    def evaluate(self, data_type='test', step=None):
        step = self.global_step if step is None else step
        loader = self.test_dataloader if data_type == 'test' else self.train_dataloader

        losses, accuracy = [], []
        nrof_samples = 0

        self.model.eval()

        for i, batch in enumerate(tqdm(loader, desc=f'Evaluate on {data_type} data')):
            with torch.no_grad():
                loss, output = self.make_training_step(batch, update=False)

            gt_words = batch[-1]
            pred_words = self.train_dataset.seq_to_word(decode_best_path(output))
            acc = np.sum([
                1 - (Levenshtein.distance(pred, gt) / max(len(pred), len(gt)))
                for pred, gt in zip(pred_words, gt_words)
            ])

            nrof_samples += len(gt_words)
            losses.append(loss * len(gt_words))
            accuracy.append(acc)

        loss = np.sum(losses) / nrof_samples
        acc = np.sum(accuracy) / nrof_samples

        print(f'\nevaluation on {data_type} set: - {nrof_samples}/{len(loader.dataset)}')
        self.console_log(step, names=['loss', 'acc'], metrics=[loss, acc])
        print()

        self.logger.log_metrics([f'eval_{data_type}/loss', f'eval_{data_type}/acc'], [loss, acc], step)

    def overfit(self):
        self.model.train()
        batch = next(iter(self.train_dataloader))
        for step in range(0, 2000):
            loss, output = self.make_training_step(batch)

            gt_words = batch[-1]
            pred_words = self.train_dataset.seq_to_word(decode_best_path(output))

            acc = np.sum([
                1 - (Levenshtein.distance(pred, gt) / max(len(pred), len(gt)))
                for pred, gt in zip(pred_words, gt_words)
            ]) / len(gt_words)

            print(f"\nstep - {step}, loss - {loss}, acc - {acc}")
            print("GT:", gt_words)
            print("PRED:", pred_words)

            # self.scheduler.step()


if __name__ == '__main__':
    trainer = Trainer(cfg=train_cfg)
    trainer.fit()
