import os
import json
import argparse
import time

from network.network import *
from utils.dataloader import CifarLoader
from utils.logger import LoggerWriter

from train import Train
from test import Test


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='config.json',
                    help='the path of global config file.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='the path of checkpoint and program will run checkpoint data.')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--mode', type=str, default='train',
                    help='the mode of app will run, plz choose among ["train", "test", "predict"]')
parser.add_argument('--device', type=str, default='cuda',
                    help='the device of app will run, plz choose among ["cuda", "cpu"]')
                    
args = parser.parse_args()

model_name = args.model_name
checkpoint = args.checkpoint
mode = args.mode
device = args.device

if mode == 'train':
    config_file = open(args.config, 'r').read()
    config = json.loads(config_file)
else:
    config_in_checkpoint = os.path.join(checkpoint, 'config', 'config.json')
    config_file = open(config_in_checkpoint, 'r').read()
    config = json.loads(config_file)
    
if 'cifar_type' in config['data_params']:
    data_loader = CifarLoader(config['data_params'])    

if mode == 'train':
    logger = LoggerWriter(config, checkpoint)
    logger.set_log_format('Epoch,Loss-CE,Acc-k1,Acc-k3,Pre,Recall,F1,AUC,Time\n')
    logger.init_logs()    
    
    net_name = config['network_params']['name']
    net = eval(net_name)(config['network_params'], device)
    
    if config['data_params']['valid_scale'] == 0:
        train_data_loader = data_loader.trainloader
        valid_data_loader = data_loader.validloader
    else:
        train_data_loader = data_loader.trainloader
        valid_data_loader = data_loader.validloader
    
    trainer = Train(config, logger, net, train_data_loader, valid_data_loader, device)
    trainer.train()
if mode == 'test':
    logger = LoggerWriter(config, checkpoint)
    net = logger.load_model(device=device, model_name=model_name)
    test_data_loader = data_loader.testloader
    
    tester = Test(config, logger, net, test_data_loader, device)
    tester.test()
    
