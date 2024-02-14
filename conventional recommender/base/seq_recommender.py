from base.recommender import Recommender
from data.sequence import Sequence
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.sampler import next_batch_sequence_for_test
import sys


class SequentialRecommender(Recommender):
    def __init__(self, conf, training_set, test_set,valid_set, **kwargs):
        super(SequentialRecommender, self).__init__(conf, training_set, test_set,valid_set, **kwargs)
        self.data = Sequence(conf, training_set, test_set)
        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.max_len = int(self.config['max_len'])
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.sav_path =  conf.__getitem__('training.set')[:-17]+'model_result/'
        self.model_name =  conf.__getitem__('model.name')
        self.valid_data = Sequence(conf, training_set, valid_set)
        self.dataset_path =  conf.__getitem__('training.set')[:-17]


    def print_model_info(self):
        super(SequentialRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (sequence number: %d, item number %d)' % (self.data.raw_seq_num,self.data.item_num))
        #print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def predict(self,seq,pos,seq_len):
        return -1

    def test(self):
        inner_data = self.valid_data
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        for n, batch in enumerate(next_batch_sequence_for_test(inner_data, self.batch_size,max_len=self.max_len)):
            seq, pos, seq_len = batch
            seq_names = [seq_full[0] for seq_full in inner_data.original_seq[n*self.batch_size:(n+1)*self.batch_size]]
            candidates = self.predict(seq, pos, seq_len)
            for name,res in zip(seq_names,candidates):
                ids, scores = find_k_largest(self.max_N, res)
                item_names = [inner_data.id2item[iid] for iid in ids if iid!=0 and iid<=inner_data.item_num]
                rec_list[name] = list(zip(item_names, scores))
            if n % 100 == 0:
                process_bar(n, inner_data.raw_seq_num/self.batch_size)
        process_bar(inner_data.raw_seq_num, inner_data.raw_seq_num)
        print('')
        return rec_list
    
    def eva_test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        for n, batch in enumerate(next_batch_sequence_for_test(self.data, self.batch_size,max_len=self.max_len)):
            seq, pos, seq_len = batch
            seq_names = [seq_full[0] for seq_full in self.data.original_seq[n*self.batch_size:(n+1)*self.batch_size]]
            candidates = self.predict(seq, pos, seq_len)
            for name,res in zip(seq_names,candidates):
                ids, scores = find_k_largest(self.max_N, res)
                item_names = [self.data.id2item[iid] for iid in ids if iid!=0 and iid<=self.data.item_num]
                rec_list[name] = list(zip(item_names, scores))
            if n % 100 == 0:
                process_bar(n, self.data.raw_seq_num/self.batch_size)
        process_bar(self.data.raw_seq_num, self.data.raw_seq_num)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('Sequence Id: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
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
        out_dir = self.output['-dir']
        file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        # FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)


        import pandas as pd
        rec_save_dict = {}
        gt_save_dict = {}

        if True:
            d = pd.read_csv(self.dataset_path+'data_aug.csv',names=['u','i'])
            du = d['u'].to_list()
            dl = []
            for i in rec_list:
                if int(i) in du:
                    tp = rec_list[i]
                    dl.append([__[0] for __ in tp[:2]])
            df1 = pd.DataFrame(dl, columns=['a','b'])
            df1.to_csv(self.dataset_path+f'data_aug_{self.model_name}.csv', index=None, header=None)

        
        for i in rec_list:
            flag = False
            gt_v = list(self.data.test_set[i].keys())[0]
            user_idd = i
            item_idd = []
            tp = rec_list[i]
            for j in tp:
                item_idd.append(j[0])
                if gt_v == j[0]:
                    flag=True

            if flag==True:
                rec_save_dict[user_idd] = item_idd
                gt_save_dict[user_idd] = [gt_v]

        
        import os
        counter = 1
        def append_file(source_file_path, target_file_path):
                with open(source_file_path, 'r') as source_file, open(target_file_path, 'a') as target_file:
                    for line in source_file:
                        target_file.write(line)

        if os.path.exists( self.sav_path+self.model_name+f'rec_save_dict{counter}.csv' ):
            counter += 1
        data_list = [[k] + v for k, v in rec_save_dict.items()]
        df = pd.DataFrame(data_list)
        df.to_csv(self.sav_path+self.model_name+f'rec_save_dict{counter}.csv', index=None, header=None)
        data_list = [[k] + v for k, v in gt_save_dict.items()]
        df = pd.DataFrame(data_list)
        df.to_csv(self.sav_path+self.model_name+f'gt_save_dict{counter}.csv', index=None, header=None)

        if  counter == 2:
            source_file_path = self.sav_path+self.model_name+f'rec_save_dict{counter}.csv'
            target_file_path = self.sav_path+self.model_name+f'rec_save_dict{counter-1}.csv'
            append_file(source_file_path, target_file_path)
            source_file_path = self.sav_path+self.model_name+f'gt_save_dict{counter}.csv'
            target_file_path = self.sav_path+self.model_name+f'gt_save_dict{counter-1}.csv'
            append_file(source_file_path, target_file_path)

        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))


    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure
