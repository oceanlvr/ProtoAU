from wandb import AlertLevel
from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys
from picture.feature import plot_features
import wandb


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(
            conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = {
            'epoch': -1,
            'metric': {},
            'addon': {},
            'hasRecord': False
        }

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (
            self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (
            self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format(
                '+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(max(self.config['ranking']), candidates)  # WIP
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append(
            'userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.config['output']
        file_name = self.config['name'] + '@' + current_time + \
            '-top-' + str(max(self.config['ranking'])) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['name'] + '@' + current_time + '-performance' + '.txt'
        rankings = [int(num) for num in self.config['ranking']]
        self.result = ranking_evaluation(self.data.test_set, rec_list, rankings)

        FileIO.write_file(out_dir, file_name, str(self.result))
        print('The result of %s:\n%s' %
              (self.config['name'], ''.join(str(self.result))))

    def update_bestPerformance(self, measure, epoch):
        maxRk = max(self.config['ranking'])
        max_rank_metric = measure[maxRk]
        count = 0
        for k in max_rank_metric:
            if maxRk not in self.bestPerformance['metric']:
                self.bestPerformance['metric'][maxRk] = {}
            if k not in self.bestPerformance['metric'][maxRk]:
                self.bestPerformance['metric'][maxRk][k] = -1
            if self.bestPerformance['metric'][maxRk][k] < max_rank_metric[k]:
                count += 1
            else:
                count -= 1

        if (not self.bestPerformance['hasRecord']) or count > 0:
            addon = {
                'user_emb': self.model.embedding_dict['user_emb'].detach().cpu().numpy(),
                'item_emb': self.model.embedding_dict['item_emb'].detach().cpu().numpy()
            }
            bestPerformance = {
                'epoch': epoch,
                'metric': measure,
                'addon': addon,
                'hasRecord': True
            }
            self.bestPerformance = bestPerformance
            logPerformance = {'epoch':epoch}
            for top in measure:
                for metric in measure[top]:
                    logPerformance['best.'+metric+'@'+str(top)] = measure[top][metric]
            wandb.log(logPerformance)
            # wandb.alert(
            #     title="Updated bestPerformance", 
            #     text=f"Epoch: f{epoch}, metrics: {max_rank_metric}",
            #     level=AlertLevel.INFO,
            #     wait_duration=300
            # )
            self.save()

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()

        # {
        #     '10': {
        #         'Hit Ratio': 0.2,
        #         'Precision': 0.2,
        #     },
        #     '50': {
        #         'Hit Ratio': 0.4,
        #         'Precision': 0.1,
        #     }
        # }
        measure = ranking_evaluation(self.data.test_set, rec_list, self.config['ranking'])

        # update best performance
        self.update_bestPerformance(measure, epoch + 1)

        max_rank_metric = measure[max(self.config['ranking'])]
        max_rank = str(max(self.config['ranking']))

        print('-' * 120)
        print('Real-Time Ranking Performance (Top-' + max_rank + ' Item Recommendation)')
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join([k+':'+str(v) for k, v in max_rank_metric.items()]))

        # set target for wandb and print log info
        log_info = {
            'epoch': epoch + 1
        }
        for topk,ind in measure.items():
            for name,value in ind.items():
                log_info[str(name)+'@'+str(topk)] = value
        wandb.log(log_info)

        # bp = ''
        # curBestPerformanceMetric = self.bestPerformance['metric']
        # bp += 'Hit Ratio' + ':' + \
        #     str(curBestPerformanceMetric['Hit Ratio']) + ' | '
        # bp += 'Precision' + ':' + \
        #     str(curBestPerformanceMetric['Precision']) + ' | '
        # bp += 'Recall' + ':' + str(curBestPerformanceMetric['Recall']) + ' | '
        # bp += 'MDCG' + ':' + str(curBestPerformanceMetric['NDCG'])
        # print('*Best Performance* ')
        # print('Epoch:', str(self.bestPerformance['epoch']) + ',', bp)
        # print('-' * 120)
        # print('Addon:', ',', str(self.bestPerformance['addon']))
        # print('-' * 120)
        # if (epoch + 1) % 10 ==0:
        #     self.drawheatmaps()
        return measure

    def afterTrain(self):
        # self.drawheatmaps()
        pass

    def drawheatmaps(self):
        drawheatmap = lambda emb, name: plot_features(emb, self.config['name'] + '_' + name)
        drawheatmap(self.bestPerformance['addon']['user_emb'], 'user_emb_'+str(self.bestPerformance['epoch']))
        drawheatmap(self.bestPerformance['addon']['item_emb'], 'item_emb_'+str(self.bestPerformance['epoch']))

    
