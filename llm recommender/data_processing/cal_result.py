import sys
import pandas as pd
import math
from math import log

dataset_names = ['ml-1M']

alpha1 = 0.5
alpha2 = 0.05
alpha3 = 0.025
for dataset_name in dataset_names:

    model_names = ['MF','LightGCN','MixGCF','SGL','SASRec','BERT4Rec', 'CL4SRec']


    if dataset_name == 'ml-100k':
        effect_num = 927

        df_like = pd.read_csv(f'./{dataset_name}/train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
        df_dislike = pd.read_csv(f'./{dataset_name}/dislike.txt', header=None, names=['u', 'i', 'r', 't'])
        movie_info = pd.read_csv(f'./{dataset_name}/movie_info.csv', header=None,
                                 names=['movie_id', 'movie_name', 'url', 'genre', 'genr0', 'genre1', 'genre2', 'genre3',
                                        'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10',
                                        'genre11', 'genre12', 'genre13', 'genre14', 'genre15', 'genre16', 'genre17',
                                        'genre18', 'genre19'], sep='|', engine='python', encoding='latin-1')
        df_like_p = pd.read_csv(f'./{dataset_name}/train_set_prediction.csv', usecols=[0, 1, 2, 3])  # pointwise
        df_like_p.columns = ['u', 'i', 'r', 't']

        movie_id_list = movie_info['movie_id'].tolist()
        movie_name_list = movie_info['movie_name'].tolist()
        movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}

    elif dataset_name == 'ml-1M':

        effect_num = 6032

        df_like = pd.read_csv(f'./{dataset_name}/train_set.csv')
        df_dislike = pd.read_csv(f'./{dataset_name}/dislike.txt', header=None, names=['u', 'i', 'r', 't'])
        movie_info = pd.read_csv(f'./{dataset_name}/movies.dat', header=None, names=['movie_id', 'movie_name', 'genre'],
                                 sep='::', engine='python', encoding='latin-1')

        df_like_p = pd.read_csv(f'./{dataset_name}/train_set_prediction.csv', usecols=[0, 1, 2, 3])  # pointwise
        df_like_p.columns = ['u', 'i', 'r', 't']

        movie_id_list = movie_info['movie_id'].tolist()
        movie_name_list = movie_info['movie_name'].tolist()
        movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}

    elif dataset_name == 'bookcrossing':
        effect_num = 1747

        df_like = pd.read_csv(f'./{dataset_name}/train_set.txt', header=None, names=['u', 'i', 'r'], sep=' ')
        df_dislike = pd.read_csv(f'./{dataset_name}/dislike.txt', header=None, names=['u', 'i', 'r'])
        movie_info = pd.read_csv(f'./{dataset_name}/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        df_like_p = pd.read_csv(f'./{dataset_name}/train_set_prediction.csv', usecols=[0, 1, 2])  # pointwise
        df_like_p.columns = ['u', 'i', 'r']

        movie_id_list = movie_info['ISBN'].tolist()
        movie_name_list = movie_info['Book-Title'].tolist()
        movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}


    def sort_list_reverse_with_indices(lst):
        sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        return sorted_indices


    for modelname in model_names:
        if modelname in ['BERT4Rec', 'CL4SRec', 'SASRec']:
            if dataset_name == 'ml-100k':
                effect_num = 940
            if dataset_name == 'ml-1M':
                effect_num = 6035
            if dataset_name == 'bookcrossing':
                effect_num = 1708

        print('-' * 15, modelname, '-' * 15)
        idd = '1'
        kk = 5
        gt_list = pd.read_csv(f'./{dataset_name}/model_result/{modelname}gt_save_dict{idd}.csv', header=None,
                              names=['u', 'i'])

        rec_list = pd.read_csv(f'./{dataset_name}/model_result/{modelname}rec_save_dict{idd}.csv', header=None,
                               names=['v' + str(i) for i in range(11)])
        test_u = rec_list['v0'].tolist()
        mov_id = gt_list['i'].tolist()

        num_correct = 0
        num_wrong = 0

        l = []
        for i in range(1, 11):
            count = 0
            templ = rec_list['v' + str(i)].tolist()

            for j in range(len(mov_id)):
                if mov_id[j] == templ[j]:
                    count += 1
            l.append(count)

        l_ = l
        l = l_[:3]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str1 = f'&{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)} &'

        l = l_[:5]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str2 = f'{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)}'

        ran_list = pd.read_csv(f'./{dataset_name}/my_test_list{modelname}.txt',
                               names=['r' + str(i) for i in range(2 * kk)])
        ran_lists = ran_list.values.tolist()

        hybird_list = []

        '''
        Pairwise Ranking
        '''
        print('Pairwise Ranking')
        with open(f'/path/to/llmres_{dataset_name}_{modelname}_pairwise.txt',
                  'r') as f:
            lines = f.readlines()

        with open(f'/path/to/llmres_{dataset_name}_{modelname}_pairwise_inv.txt',
                  'r') as f:
            lines2 = f.readlines()

        ll = [0] * 10

        for idx, row in rec_list.iterrows():

            w = test_u[idx] * 200

            scores = [kk * 2 - i for i in range(2 * kk)]
            local_ran_lists = ran_lists[idx]
            local_hy_list = [0] * 10
            for ii in range(kk):
                local_idx = kk * idx + ii

                fir_idx = int(local_ran_lists[ii])
                # print(local_ran_lists)
                sec_idx = int(local_ran_lists[ii + kk])

                try:
                    llmtag = lines[local_idx]
                except Exception:
                    print( len(lines), local_idx)

                if 'Yes' in llmtag:
                    scores[fir_idx] += w
                else:
                    scores[sec_idx] += w

            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])[::-1]
            row_list = row.tolist()[1:]
            sorted_rows = [row_list[iii] for iii in sorted_indices]

            if mov_id[idx] in sorted_rows[:kk]:
                num_correct += 1
            else:
                num_wrong += 1

            if mov_id[idx] in sorted_rows:
                idx_ = sorted_rows.index(mov_id[idx])
            else:
                idx_ = 100
            ll[idx_] += 1

        num_correct = 0
        num_wrong = 0

        l_ = ll
        l = l_[:3]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str1 = f'&{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)} &'

        l = l_[:5]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str2 = f'{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)}'

        '''
        Pairwise Inverse
        '''
        print('Pairwise Inverse')
        # ll=[]
        ll = [0] * 10
        for idx, row in rec_list.iterrows():

            w = test_u[idx] * 200

            scores = [kk * 2 - i for i in range(2 * kk)]
            local_ran_lists = ran_lists[idx]
            local_hy_list = [0] * 10
            for ii in range(kk):
                local_idx = kk * idx + ii

                fir_idx = int(local_ran_lists[ii])
                sec_idx = int(local_ran_lists[ii + kk])

                try:
                    llmtag = lines[local_idx]
                    llmtag2 = lines2[local_idx]

                except Exception:
                    print( len(lines), local_idx)

                if 'Yes' in llmtag and 'No' in llmtag2:
                    scores[fir_idx] += w
                    local_hy_list[fir_idx] += alpha1
                elif 'Yes' in llmtag2 and 'No' in llmtag:
                    # print(sec_idx)
                    scores[sec_idx] += w
                    local_hy_list[sec_idx] += alpha1
                else:
                    scores[sec_idx] += (w / 2)
                    local_hy_list[sec_idx] += (alpha1 / 2)
                    local_hy_list[fir_idx] += (alpha1 / 2)

            hybird_list.append(local_hy_list)

            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])[::-1]
            row_list = row.tolist()[1:]
            sorted_rows = [row_list[iii] for iii in sorted_indices]

            if mov_id[idx] in sorted_rows[:kk]:
                num_correct += 1
            else:
                num_wrong += 1

            if mov_id[idx] in sorted_rows:
                idx_ = sorted_rows.index(mov_id[idx])
            else:
                idx_ = 11
            ll[idx_] += 1


        num_correct = 0
        num_wrong = 0

        l_ = ll
        l = l_[:3]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str1 = f'&{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)} &'

        l = l_[:5]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str2 = f'{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)}'


        pair_str = str1 + str2

        '''
        Pointwise Ranking
        '''
        print('Pointwise Ranking')
        with open(f'/path/to/llmres_{dataset_name}_{modelname}_pointwise.txt',
                  'r') as f:
            lines = f.readlines()

        ll = [0] * 10

        num_correct = 0
        num_wrong = 0
        for idx, row in rec_list.iterrows():

            local_score = []

            for ii in range(10):
                try:
                    s_ = int(lines[idx * 10 + ii].split(':')[-1].replace(' ', ''))
                except Exception:
                    s_ = 3
                local_score.append(s_ + alpha2 * (10 - ii))

            hybird_list[idx] = [hybird_list[idx][_] + local_score[_] for _ in range(len(hybird_list[idx]))]

            res_idx_list = sort_list_reverse_with_indices(local_score)

            row = row.tolist()[1:]

            local_ran_lists = row
            sorted_rows = [local_ran_lists[__] for __ in res_idx_list]

            if mov_id[idx] in sorted_rows[:kk]:
                num_correct += 1
            else:
                num_wrong += 1

            if mov_id[idx] in sorted_rows:
                idx_ = sorted_rows.index(mov_id[idx])
            else:
                idx_ = 11
                print(idx, mov_id[idx], sorted_rows, row)

            ll[idx_] += 1

        num_correct = 0
        num_wrong = 0

        l_ = ll
        l = l_[:3]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str1 = f'&{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)} &'

        l = l_[:5]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str2 = f'{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)}'
        print(str1 + str2)

        point_str = str1 + str2

        '''
        Listwise Ranking
        '''
        print('Listwise Ranking')
        with open(f'/path/to/llmres_{dataset_name}_{modelname}_listwise.txt',
                  'r') as f:
            lines = f.readlines()

        ll = [0] * 10

        num_correct = 0
        num_wrong = 0

        for idx, row in enumerate(mov_id):
            mnn = movie_name_dict[row]

            if mnn in lines[idx]:
                num_correct += 1
            else:
                num_wrong += 1
                ll[8] += 1

            line_lst = lines[idx].split('"",')
            line_lst = line_lst[:10]
            for idx_ in range(len(line_lst)):
                if mnn in line_lst[idx_]:
                    ll[idx_] += 1

            local_score = [0] * 10
            row = rec_list.iloc[idx][1:]
            for __ in range(len(row)):
                if movie_name_dict[row[__]] in lines[idx]:
                    for idx_ in range(len(line_lst)):
                        if movie_name_dict[row[__]] in line_lst[idx_]:
                            local_score[__] += (5 - idx_)

            hybird_list[idx] = [hybird_list[idx][_] + alpha3 * local_score[_] for _ in range(len(hybird_list[idx]))]


        num_correct = 0
        num_wrong = 0

        l_ = ll
        l = l_[:3]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str1 = f'&{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)} &'

        l = l_[:5]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str2 = f'{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)}'

        list_str = str1 + str2

        '''
        Hybird ranking
        '''
        print('Hybird ranking')
        ll = [0] * 10

        num_correct = 0
        num_wrong = 0
        for idx, row in rec_list.iterrows():

            res_idx_list = sort_list_reverse_with_indices(hybird_list[idx])

            row = row.tolist()[1:]

            local_ran_lists = row
            sorted_rows = [local_ran_lists[__] for __ in res_idx_list]

            if mov_id[idx] in sorted_rows[:kk]:
                num_correct += 1
            else:
                num_wrong += 1

            if mov_id[idx] in sorted_rows:
                idx_ = sorted_rows.index(mov_id[idx])
            else:
                idx_ = 11
            ll[idx_] += 1

        num_correct = 0
        num_wrong = 0

        l_ = ll
        l = l_[:3]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str1 = f'&{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)} &'

        l = l_[:5]
        cu_s = 0
        for i in range(len(l)):
            cu_s += l[i] * 1 / math.log(2 + i, 2)

        str2 = f'{round(sum(l) / (effect_num * 5), 4)} & {round(cu_s / (effect_num * 5), 4)}'
        print(str1 + str2)

        hybird_str = str1 + str2
        print(point_str,f'\n{pair_str}\n{list_str}\n{hybird_str}')
        print('-' * 34)




