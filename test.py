import os
import time
import numpy as np
from utils.utils import format_runtime, ClassMetrics

import torch 
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F

from network.loss import LossUtils

class Test:
    def __init__(self, config, logger, net, test_data_loader, device):
        self.config = config
        self.device = device
        
        self.logger = logger
        self.net = net
        
        # for param_tensor in self.net.state_dict():
        #     print(param_tensor, "\t", self.net.state_dict()[param_tensor].size())
            
        self.test_data_loader = test_data_loader
        self.num_classes = config['data_params']['num_classes']
        
        Loss = LossUtils(self.device)
        self.loss = Loss(config['train_params']['loss'])        
        
        self.metrics = ClassMetrics(self.num_classes, 'macro')
        self.metrics.set_report_metrics(config['train_params']['report_metrics'])
        self.report_format = config['train_params']['report_format']
                
    
    def report_save(self, epoch, train_report, valid_report, step_time, save_metric, mode):
        self.logger.save_model(self.net, self.val_acc_1, mode=self.save_mode)
        report_data = [epoch]
        for x, y in zip(train_report, valid_report):
            report_data.append(x)
            report_data.append(y)
        report_data.append(step_time)
        
        print(self.report_format.format(*report_data))
            
            
    def test(self):
        test_loss = []
        self.net.eval()
        step_time = time.time()
        true_report, pred_report = None, None
        for idx, (data, labels) in enumerate(self.test_data_loader):
            x_batch = Variable(torch.FloatTensor(data).to(self.device), requires_grad=False)
            y_batch = Variable(labels.to(self.device), requires_grad=False)
            
            output = self.net(x_batch)
            loss = self.loss(output, y_batch)
            
            test_loss.append(loss.item())
            
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
        # for name, data in zip(self.config['train_params']['report_metrics'], all_report):
        #     print('{}: {}'.format(name, data))
            
        for name in self.config['train_params']['report_metrics']:
            print(name, end='\t')
        print('')
        for data in all_report:
            print('{:.4}'.format(data), end='\t')
        print('')
        print(true_report.shape)
        print(pred_report.shape)
        return all_report

        
            
