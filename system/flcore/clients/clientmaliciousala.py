import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.clients.clientavg import clientAVG
from flcore.clients.clientala import clientALA

from utils.privacy import *

from flcore.attack.attack import *


class ClientMaliciousALA(clientALA):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.rate_client_fake = args.rate_client_fake
        self.atack = args.atack

        self.list_global_model = []
        self.delay_atk = args.delay_atk
        self.round_init_atk = args.round_init_atk
        self.model = copy.deepcopy(args.model)

    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def set_parameters(self, model):
        self.list_global_model.append(model)
        return super().set_parameters(model)
    
    def send_local_model(self):

        self.is_malicious = np.random.choice([False, True], 
                                     p = [1 - self.rate_client_fake, self.rate_client_fake])
        if self.is_malicious:
            
            print(f'malicioso: {self.id}')
            if self.atack == 'zero':
                return model_zeros(self.model, self.device)
            elif self.atack == 'random':
                return random_param(self.model, self.device)
            elif self.atack == 'shuffle':
                return shuffle_model(self.model)

        return self.model