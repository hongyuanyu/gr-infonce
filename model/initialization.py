import os
from copy import deepcopy
import numpy as np
from .data import load_data
from .model import Model
from .model_contrastive import ModelContrastive
from .model_mix import ModelMix
from .model_usl import ModelUSL
from pdb import set_trace

def initialize_data(config):
    print("Initializing data source...")
    train_source, test_source = load_data(config)
    print("Data initialization complete.")
    return train_source, test_source

def initialize_model(config):
    print("Initializing model...")
    
    if config['contrastivelearning']:
        m = ModelContrastive(config)
    elif config['model_mix']:
        m = ModelMix(config)
    #elif  config['self_supervised_weight']:
    #    m = ModelUSL(config)
    else:
        m = Model(config)
    print("Model initialization complete.")
    return m

def initialization(config):
    print("Initialzing...")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    if config['model_mix']:
        config['pid_num'] = 73+43
        train_source_all, _ = initialize_data(config)
        config['train_source_all'] = train_source_all

        config['dataset_path'] = config['dataset_path_clean']
        config['pid_num'] = 30
        train_source_clean, test_source = initialize_data(config)
        config['train_source_clean'] = train_source_clean
        config['test_source'] = test_source
        config['save_name'] = '{}_{}_{}_{}'.format(config['model_name'], config['dataset'], config['pid_num'], config['pid_shuffle'])
        
        config['dataset_path'] = config['dataset_path_noisy']
        config['pid_num'] = -1
        train_source_noisy, _ = initialize_data(config)
        config['train_source_noisy'] = train_source_noisy
                
        model = initialize_model(config)
    else:
        config['pid_num'] = 73+43
        train_source, test_source = initialize_data(config)
        config['train_source'] = train_source
        config['test_source'] = test_source
        config['save_name'] = '{}_{}_{}_{}'.format(config['model_name'], config['dataset'], config['pid_num'], config['pid_shuffle'])
        config.update({'num_id_all': len(config['train_source'].label_set)})
        model = initialize_model(config)
    return model