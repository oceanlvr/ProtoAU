import yaml
from yaml import SafeLoader
import wandb
from util.helper import fix_random_seed, composePath, mergeDict
import argparse

from data.loader import FileIO
from util.helper import composePath

class Runner(object):
    def __init__(self, config):
        self.config = config
        self.training_data = []
        self.test_data = []
        self.load_dataset()
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.' + self.config['type'] + '.' + \
            self.config['name'] + ' import ' + self.config['name']
        exec(import_str)
        recommender = self.config['name'] + \
            '(self.config,self.training_data,self.test_data)'
        eval(recommender).execute()

    def load_dataset(self):
        train_data_path = composePath(
            './dataset', self.config['dataset'], 'train.txt')
        test_data_path = composePath(
            './dataset', self.config['dataset'], 'test.txt')
        self.training_data = FileIO.load_data_set(
            train_data_path,
            self.config['type']
        )
        self.test_data = FileIO.load_data_set(
            test_data_path,
            self.config['type']
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ProtoAU')
    parser.add_argument('--dataset', type=str, default='Yelp')
    parser.add_argument('--root', type=str, default='/workspace/')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--tags', nargs='*', default=[]) 
    parser.add_argument('--group', type=str, default='default')  #
    parser.add_argument('--job_type', type=str, default='eval')  #
    parser.add_argument('--notes', type=str)
    parser.add_argument('--run_name', type=str)
    # graph args
    parser.add_argument('--ranking', nargs='*', type=int)
    parser.add_argument('--embedding_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lambda', type=float)

    parser.add_argument('--model_config.hyper_layers', type=int)
    parser.add_argument('--model_config.ssl_reg', type=float)
    parser.add_argument('--model_config.proto_reg', type=float)
    parser.add_argument('--model_config.alpha', type=float)
    parser.add_argument('--model_config.num_clusters', type=float)
    parser.add_argument('--model_config.eps', type=float)
    parser.add_argument('--model_config.tau', type=float)

    sequential_models = []
    args = vars(parser.parse_args())

    config_path = composePath(args['root'], 'conf', args['model'] + '.yaml')
    config = yaml.load(open(config_path), Loader=SafeLoader)[args['dataset']]
    config = mergeDict(config, args)
    run = wandb.init(project="ProtoAU", group=args['group'], job_type=args['job_type'], entity="your_name", name=args['run_name'] or None, config=config)

    # here is your random seed
    # fix_random_seed(wandb.config['seed'])
    print('='*10, 'wandb.config', '='*10)
    print(wandb.config)
    print('='*10, 'wandb.config', '='*10)

    rec = Runner(wandb.config)
    rec.execute()
    run.finish()
