import os
import time
import json
from shutil import copyfile, rmtree
from PIL import Image
import importlib

import numpy as np


from network.network import *

EPSLION = 1e-10

class LoggerWriter:
    '''
    LoggerWriter completes the functions implementation of log writing and model saving
    Inputs: config, checkpoint
        - config: the global config file for whole application
        - checkpoint: the checkpoint path to load, default is None
    '''
    def __init__(self, config, checkpoint=None):
        self.config = config
        self.checkpoint = checkpoint
        self.model_save_index = 0
        self.last_metric = {}

        self.net_name = self.config['network_params']['name']
        self.lr_name = self.config['train_params']['learning_rate']
        self.loss_name = self.config['train_params']['loss']

        self.proj_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.init_path()
        self.set_log_format()


    def init_path(self):
        '''
        init path based on checkpoint path, if it is None, init path based on time, network's name, loss's name, and lr
        '''
        if self.checkpoint is None:
            log_root = self.config['log_params']['log_root']
            if not os.path.exists(log_root):
                raise RuntimeError('Log root directory "{}" does not exist'.format(log_root))
            create_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


            self.log_dir = os.path.join(self.config['log_params']['log_root'], '{}_{}_{}_{}'.format(
                create_time, self.net_name, self.lr_name, self.loss_name))

            os.mkdir(self.log_dir)

            self.config_save_path = os.path.join(self.log_dir, 'config')
            self.weight_save_path = os.path.join(self.log_dir, 'weight')
            self.model_save_path = os.path.join(self.log_dir, 'model')
            self.loss_save_path = os.path.join(self.log_dir, 'loss')


            os.mkdir(self.config_save_path)
            os.mkdir(self.weight_save_path)
            os.mkdir(self.model_save_path)
            os.mkdir(self.loss_save_path)


            save_config_file = open(os.path.join(self.config_save_path, 'config.json'), 'w')
            json.dump(self.config, save_config_file, indent=4)
            save_config_file.close()

            copyfile(os.path.join(self.proj_root, 'network/network.py'), os.path.join(self.model_save_path, 'network.py'))
            copyfile(os.path.join(self.proj_root, 'network/loss.py'), os.path.join(self.model_save_path, 'loss.py'))
            copyfile(os.path.join(self.proj_root, 'train.py'), os.path.join(self.model_save_path, 'train.py'))

        else:
            if not os.path.exists(self.checkpoint):
                raise RuntimeError('Checkpoint directory "{}" does not exist'.format(self.checkpoint))
            self.log_dir = self.checkpoint

            self.config_save_path = os.path.join(self.log_dir, 'config')
            self.weight_save_path = os.path.join(self.log_dir, 'weight')
            self.model_save_path = os.path.join(self.log_dir, 'model')
            self.loss_save_path = os.path.join(self.log_dir, 'loss')


    def set_log_format(self, log_header=None):
        '''
        This function sets the table header of log file, if log_header is None, set as default format
        '''
        if log_header is None:
            self.log_header = 'Epoch,Iter,Loss-{},Time\n'.format(self.loss_name)
            self.log_format = '{},{},{},{}\n'
        else:
            self.log_header = log_header
            self.log_format = ','.join(['{}']*len(self.log_header.split(',')))+'\n'

    def init_logs(self):
        '''
        Create log file
        '''
        self.train_log = os.path.join(self.loss_save_path, 'train_loss.csv')
        self.valid_log = os.path.join(self.loss_save_path, 'valid_loss.csv')
        if not os.path.exists(self.train_log):
            with open(self.train_log, 'w') as f:
                f.write(self.log_header)
                f.close()
        if not os.path.exists(self.valid_log):
            with open(self.valid_log, 'w') as f:
                f.write(self.log_header)
                f.close()

    def write_train_log(self, args):
        with open(self.train_log, 'a') as f:
            f.write(self.log_format.format(*args))
            f.close()
    def write_valid_log(self, args):
        with open(self.valid_log, 'a') as f:
            f.write(self.log_format.format(*args))
            f.close()

    def load_model(self, model_name=None, device='cuda'):
        '''
        Load saved model based on the network and weight in checkpoint path
        '''
        # net = eval(self.net_name)(self.config['network_params'], device)  # load model based on network name in config
        
        # load network model from checkpoint file
        spec = importlib.util.spec_from_file_location(
            'network',
            os.path.join(self.checkpoint, 'model', 'network.py')
        )
        load_network_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_network_module)
        net = eval('load_network_module.' + self.net_name)(self.config['network_params'], device)
        
        if model_name is not None:
            model_path = os.path.join(self.weight_save_path, model_name + '.pkl')
            self.model_name = model_name
        elif os.path.exists(os.path.join(self.weight_save_path, 'best_model.pkl')):
            # if model_name is None, load best_model.pkl as default weight
            model_path = os.path.join(self.weight_save_path, 'best_model.pkl')
            self.model_name = 'best'
        else:
            raise RuntimeError('The model "{}" dose not exist'.format(model_name))

        net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        return net

    def save_model(self, net, metric, mode='min', prefix=None):
        '''
        Save the weight of model
        Paramters:
            - net: network<torch.nn.Module>
            - metric: the evaluation metrics which the model saving is based on
            - mode: mode limited in ['min', 'max'], if mode is 'min', select the minimal metrics as best model
        '''
        if prefix is None:
            model_name = 'model'
        else:
            model_name = prefix + '_model'
        # torch.save(net.state_dict(), os.path.join(self.weight_save_path, '{}_{}.pkl'.format(model_name, self.model_save_index)))
        self.model_save_index += 1
        if prefix not in self.last_metric:
            torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}.pkl'.format(model_name)))
            self.last_metric[prefix] = metric
            torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}_{}00.pkl'.format(model_name, self.model_save_index // 100)))
        else:
            if mode == 'min':
                if metric < self.last_metric[prefix]:
                    torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}.pkl'.format(model_name)))
                    self.last_metric[prefix] = metric
                    torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}_{}00.pkl'.format(model_name, self.model_save_index // 100)))
            elif mode == 'max':
                if metric > self.last_metric[prefix]:
                    torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}.pkl'.format(model_name)))
                    self.last_metric[prefix] = metric
                    torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}_{}00.pkl'.format(model_name, self.model_save_index // 100)))
            else:
                raise ValueError('Save mode must be in ["max", "min"], error {}'.format(mode))

            


            





