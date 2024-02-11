from data.data import Data
from time import strftime, localtime, time
import torch

class Recommender(object):
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)
        self.current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.result = []
        self.recOutput = []
        if torch.cuda.is_available() and self.config["gpu_id"] >= 0:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config["gpu_id"])
        else:
            self.device = 'cpu'

    def initializing_log(self):
        pass

    def print_model_info(self):
        pass

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass
    
    def afterTrain(self):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        self.afterTrain()
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)
