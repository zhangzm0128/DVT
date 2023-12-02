import os
import time
import numpy as np
import math
from utils.utils import format_runtime, ClassMetrics

import torch 
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

from network.loss import LossUtils



class Train:
    def __init__(self, config, logger, net, train_data_loader, valid_data_loader, device):
        self.config = config
        self.device = device
        
        self.logger = logger
        self.net = net
        
        for param_tensor in self.net.state_dict():
            print(param_tensor, "\t", self.net.state_dict()[param_tensor].size())
            
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.num_classes = config['data_params']['num_classes']
        
        Loss = LossUtils(self.device)
        self.loss = Loss(config['train_params']['loss'])
        self.lr = config['train_params']['learning_rate']
        self.opt = config['train_params']['optimizer']
        
        self.epoch = config['train_params']['epoch']
        self.save_mode = config['train_params']['save_mode']
        
        self.metrics = ClassMetrics(self.num_classes)
        self.metrics.set_report_metrics(config['train_params']['report_metrics'])
        self.report_format = config['train_params']['report_format']
        
        

        self.no_improve = 0
        self.stopper = False
        self.best_val_loss = None
        self.set_opt()
        self.set_scheduler()
        
    def set_opt(self):
        if 'opt_args' not in self.config['train_params']:
            self.opt = eval('optim.' + self.opt)(self.net.parameters(), lr=self.lr)
            # self.opt = eval('optim.' + self.opt)(self.net.parameters(), lr=self.lr)
            # self.opt = eval('optim.' + self.opt)(
            #     [{'params': self.net.parameters(), 'lr': self.lr*10}, 
            #      # {'params': self.net.transformer.parameters(), 'lr': self.lr},
            #     ], lr=self.lr)
        else:
            self.opt = eval('optim.' + self.opt)(self.net.parameters(), self.lr, **self.config['train_params']['opt_args'])
            
    def set_scheduler(self):
        if 'lr_scheduler' in self.config['train_params'] and self.config['train_params']['lr_scheduler'] != {}:
            n_iter_per_epoch = len(self.train_data_loader)
            num_steps = int(self.epoch * n_iter_per_epoch)
            warmup_steps = int(self.config['train_params']['lr_scheduler']['warmup'] * n_iter_per_epoch)
            
            self.lr_scheduler = CosineAnnealingWarmupRestarts(
                self.opt,
                first_cycle_steps=num_steps,
                cycle_mult=1.,
                max_lr = self.lr,
                min_lr = 1e-6,
                warmup_steps=warmup_steps)
        else:
            self.lr_scheduler = None
            
            
    
    def report_save(self, epoch, train_report, valid_report, step_time, save_metric, mode):
        self.logger.save_model(self.net, self.val_acc_1, mode=self.save_mode)
        report_data = [epoch]
        for x, y in zip(train_report, valid_report):
            report_data.append(x)
            report_data.append(y)
        report_data.append(step_time)
        
        print(self.report_format.format(*report_data))
        
            
    def train(self):
        # Training process
        
        epoch_num_batch = len(self.train_data_loader)
        
        train_loss = []
        
        for current_epoch in range(self.epoch):
            step_time = time.time()
            self.net.train()
            true_report, pred_report = None, None
            for idx, (data, labels) in enumerate(self.train_data_loader):
                # datas [b, n, h, w], labels (int label) [b]
                x_batch = Variable(torch.FloatTensor(data).to(self.device), requires_grad=False)
                y_batch = Variable(labels.to(self.device), requires_grad=False)
                
                self.opt.zero_grad()
                output = self.net(x_batch) # one_hot pred [b, num_classes]
                loss = self.loss(output, y_batch)
                loss.backward()
                self.opt.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                train_loss.append(loss.item())
                if true_report is None and pred_report is None:
                    true_report = labels.cpu().detach().numpy()
                    pred_report = output.cpu().detach().numpy()
                else:
                    true_report = np.r_[true_report, labels.cpu().detach().numpy()]
                    pred_report = np.r_[pred_report, output.cpu().detach().numpy()]
                
                    
            all_report = self.metrics.report(true_report, pred_report)
            step_time = time.time() - step_time
            # acc-1, acc-3, pre, recall, f1, AUC        
            # print('Train Epoch: {} -- Loss: {:.4} -- Acc-1: {:.0%} -- AUC: {:.0%} -- Time: {}'.format(
            #     current_epoch, np.mean(train_loss), all_report[0], all_report[5], format_runtime(step_time)))
                
            train_log_data = [current_epoch, np.mean(train_loss)] + all_report + [step_time]
            valid_log_data = self.valid(current_epoch)

            self.logger.write_train_log(train_log_data)
            self.logger.write_valid_log(valid_log_data)
            
            self.report_save(current_epoch, 
                [train_log_data[1], train_log_data[2], train_log_data[-2]],
                [valid_log_data[1], valid_log_data[2], valid_log_data[-2]],
                format_runtime(train_log_data[-1] + valid_log_data[-1]),
                self.val_acc_1, self.save_mode)
            
            
            # self.logger.save_model(self.net, self.val_acc_1, mode=self.save_mode)
            
    def valid(self, current_epoch):
        valid_loss = []
        self.net.eval()
        step_time = time.time()
        true_report, pred_report = None, None
        for idx, (data, labels) in enumerate(self.valid_data_loader):
            x_batch = Variable(torch.FloatTensor(data).to(self.device), requires_grad=False)
            y_batch = Variable(labels.to(self.device), requires_grad=False)
            
            output = self.net(x_batch)
            loss = self.loss(output, y_batch)
            
            valid_loss.append(loss.item())
            
            if true_report is None and pred_report is None:
                true_report = labels.cpu().detach().numpy()
                pred_report = output.cpu().detach().numpy()
            else:
                true_report = np.r_[true_report, labels.cpu().detach().numpy()]
                pred_report = np.r_[pred_report, output.cpu().detach().numpy()]
        
        all_report = self.metrics.report(true_report, pred_report)
        step_time = time.time() - step_time
        
        # print('Valid Epoch: {} -- Loss: {:.4} -- Acc-1: {:.0%} -- AUC: {:.0%} -- Time: {}'.format(
        #         current_epoch, np.mean(valid_loss), all_report[0], all_report[5], format_runtime(step_time)))
        log_data = [current_epoch, np.mean(valid_loss)] + all_report + [step_time]

        self.val_acc_1 = all_report[0]
        
        return log_data
        
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr            
