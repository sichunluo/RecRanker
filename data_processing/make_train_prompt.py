import sys
import random
import pandas as pd
import jsonlines
import numpy as np
import copy

random.seed(42)
np.random.seed(42)

dataset_names = ['ml-100k', 'bookcrossing', 'ml-1M']

isHint = True
sample_method = 'uniform'  # importance

for dataset_name in dataset_names:
    if dataset_name == 'ml-100k':
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
        df_like = pd.read_csv(f'./{dataset_name}/train_set.txt', header=None, names=['u', 'i', 'r'], sep=' ')
        df_dislike = pd.read_csv(f'./{dataset_name}/dislike.txt', header=None, names=['u', 'i', 'r'])
        movie_info = pd.read_csv(f'./{dataset_name}/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        df_like_p = pd.read_csv(f'./{dataset_name}/train_set_prediction.csv', usecols=[0, 1, 2])  # pointwise
        df_like_p.columns = ['u', 'i', 'r']

        movie_id_list = movie_info['ISBN'].tolist()
        movie_name_list = movie_info['Book-Title'].tolist()
        movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}


    if 'book' in dataset_name:
        kws = 'book'
    else:
        kws = 'movie'


    def sort_list_reverse_with_indices(lst):
        sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        return sorted_indices


    import pickle

    with open(f'../model_res_pkl/{dataset_name}_item.pkl', "rb") as file:
        cm_item = pickle.load(file)
    with open(f'../model_res_pkl/{dataset_name}_user.pkl', "rb") as file:
        cm_user = pickle.load(file)
    with open(f'../model_res_pkl/{dataset_name}_pred.pkl', "rb") as file:
        cm_pred = pickle.load(file)

    with open(f'../model_res_pkl/{dataset_name}_user_emb.pkl', "rb") as file:
        cm_user_emb = pickle.load(file)

    with open(f'./{dataset_name}/item_id_mapping.pkl', "rb") as file:
        mf_item = pickle.load(file)
    with open(f'./{dataset_name}/user_id_mapping.pkl', "rb") as file:
        mf_user = pickle.load(file)
    with open(f'./{dataset_name}/rating_matrix.pkl', "rb") as file:
        mf_pred = pickle.load(file)

    mes_list = []
    gt_list = []


    if 'ml-1M' in dataset_name:
        sample_n = 10000
    else:
        sample_n = 5000
    user_list = list(df_like['u'].unique())
    sample_list = []
    import math

    weights = [math.log(len(df_like[df_like['u'] == uni])) for uni in user_list]

    if sample_method == 'uniform':
        for i in range(sample_n):
            sample_ = random.sample(user_list, 1)[0]
            sample_list.append(sample_)
    else:
        sample_list1 = []
        sample_list2 = []

        sample_imp = int(sample_n * 0.6)
        for i in range(sample_imp):
            sample_ = random.choices(user_list, weights, k=1)[0]
            sample_list1.append(sample_)

        from sklearn.cluster import KMeans


        kmeans = KMeans(n_clusters=10, random_state=0).fit(cm_user_emb)
        labels = kmeans.labels_

        counts = np.bincount(labels)

        samples_per_cluster = np.round(counts / counts.sum() * sample_imp).astype(int)

        sampled_ids = []
        for cluster_id, samples in enumerate(samples_per_cluster):
            cluster_ids = np.where(labels == cluster_id)[0]
            sampled_ids.extend(np.random.choice(cluster_ids, samples, replace=True))

        mf_user_i = {value: key for key, value in mf_user.items()}
        sampled_ids = [mf_user_i[_] for _ in sampled_ids]

        sample_list1.extend(sampled_ids)
        from collections import Counter

        occurrences = Counter(sample_list1)
        t_occurrences = {element: 0.95 ** (count - 1) for element, count in occurrences.items()}
        sample_list2 = [t_occurrences[_] for _ in sample_list1]

        sample_list = random.choices(sample_list1, weights=sample_list2, k=sample_n)


    for uni in sample_list:
        df = df_like[df_like['u'] == uni]
        df_un = df_dislike[df_dislike['u'] == uni]

        if len(df) > 1:

            '''
            Pointwise Ranking
            '''
            dfp = df_like_p[df_like_p['u'] == uni]
            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()

            rndl = [i_ for i_ in range(len(my_list))]
            random.shuffle(rndl)
            if not 'book' in dataset_name:
                my_list = [int(my_list[x]) for x in rndl]
                my_list_r = [int(my_list_r[x]) for x in rndl]
            else:
                my_list = [(my_list[x]) for x in rndl]
                my_list_r = [(my_list_r[x]) for x in rndl]

            if len(dfp) > 50:
                topk = 50
            else:
                topk = len(dfp) - 3
            trainlist = my_list[:topk]
            trainlist_r = my_list_r[:topk]

            testlist = my_list[-1:]
            testlist_r = my_list_r[-1:]

            try:
                yy = mf_item[(testlist[0])]
                uu = mf_user[(uni)]
                mf_lable = mf_pred[uu][yy]
            except Exception:
                mf_lable = 'Unknown.'
                print(1)

            historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]
            answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]

            historical_interactions = [historical_interactions[i_] + ': ' + str(trainlist_r[i_]) + ';' for i_ in
                                       range(len(historical_interactions))]

            historical_interactions = ' '.join(historical_interactions)
            if 'book' in dataset_name:
                highest_score = 5
            else:
                highest_score = 10
            instruct0 = f'''You are a {kws} recommender system. Your task is to predict the relevance score to a target {kws} based on the user's historical {kws} ratings. The score should be between 1 and {highest_score}, where 1 is the lowest affinity and {highest_score} is the highest. Respond only with a number between 1 to {highest_score}.\n\n'''

            instruct1 = f'''User's historical {kws} ratings: <historical interactions>. \n\nQuestion: Based on the user's historical ratings, predict the relavance score of the target {kws} <movie> with the user.\n'''
            instruct2 = '''Hint: Another recommender system suggests the answer is <mf_prediction>"'''
            if isHint:
                instruct1 = instruct1 + instruct2
            instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<movie>',
                                                                                                        answer_items_set[
                                                                                                            0]).replace(
                '<mf_prediction>', str(mf_lable)[:3])

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1}]},
                {"role": "assistant", "content": [{"type": "text", "content": 'Answer: ' + str(testlist_r[0])}]}
            ]}
            mes_list.append(fi)

            '''
            Pairwise Ranking
            '''
            unlikelist = []
            if len(df_un) > 0:
                my_list = df_un['i'].tolist()
                random.shuffle(my_list)

                if not 'book' in dataset_name:
                    my_list = [int(x) for x in my_list]
                else:
                    my_list = [(x) for x in my_list]

                unlikelist = my_list[:10]

            my_list = df['i'].tolist()
            random.shuffle(my_list)

            if not 'book' in dataset_name:
                my_list = [int(x) for x in my_list]
            else:
                my_list = [(x) for x in my_list]

            if len(df) > 55:
                topk = 50
            else:
                topk = len(df) - 3
            trainlist = my_list[:topk]
            testlist = my_list[-1:]
            neglist = []
            while len(neglist) < 1:
                rn = random.sample(movie_id_list, 1)[0]
                if not rn in my_list:
                    neglist.append(rn)

            random_n = (random.random() > 0.5)

            historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]
            false_items_set = [f'"{movie_name_dict[i]}"' for i in neglist]
            answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]

            try:
                xx = cm_item[str(neglist[0])]
                yy = cm_item[str(testlist[0])]
                uu = cm_user[str(uni)]
                if cm_pred[uu][yy] > cm_pred[uu][xx]:
                    cm_lable = 'Yes.'
                else:
                    cm_lable = 'No.'
            except Exception:
                cm_lable = 'Unknown.'

            unlikelist = [x_ for x_ in unlikelist if x_ in movie_name_dict.keys()]

            user_unpre = [f'"{movie_name_dict[i]}"' for i in unlikelist]

            if len(unlikelist) < 3:
                user_unpre = 'None.'
            else:
                user_unpre = ', '.join(user_unpre)

            historical_interactions = ', '.join(historical_interactions)
            gt_list.append(movie_name_dict[testlist[0]])

            if random_n:
                first_name = answer_items_set[0]
                second_name = false_items_set[0]
                tg = 'Yes.'
            else:
                first_name = false_items_set[0]
                second_name = answer_items_set[0]
                tg = 'No.'

                if cm_lable == 'Yes.':
                    cm_lable = 'No.'
                elif cm_lable == 'No.':
                    cm_lable = 'Yes.'

            instruct0 = f'''You are a {kws} recommender system. Based on a user's likes and dislikes, determine if they would prefer one {kws} over another. Respond only with "Yes." or "No.".\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: Would the user prefer <movie1> over <movie2>?\n'''
            instruct2 = '''Hint: Another recommender system suggests the answer is "<cm_result>"'''
            if isHint:
                instruct1 = instruct1 + instruct2
            instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<user_unpre>',
                                                                                                        user_unpre).replace(
                '<movie1>', first_name).replace('<movie2>', second_name).replace('<cm_result>', cm_lable)

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1}]},
                {"role": "assistant", "content": [{"type": "text", "content": 'Answer: ' + tg}]}
            ]}
            mes_list.append(fi)

            '''
            Listwise Ranking
            '''

            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()
            if not 'book' in dataset_name:
                my_list = [int(my_list[x]) for x in rndl]
                my_list_r = [int(my_list_r[x]) for x in rndl]
            else:
                my_list = [(my_list[x]) for x in rndl]
                my_list_r = [(my_list_r[x]) for x in rndl]

            if not 'book' in dataset_name:
                largest_rating = 5
                second_largest_rating = 4
            else:
                largest_rating = 10
                second_largest_rating = 9

            num_5 = my_list_r.count(largest_rating)
            num_4 = my_list_r.count(second_largest_rating)

            if (not (num_5 > 3 and num_4 > 2)) and 'book' in dataset_name:
                largest_rating = 9
                second_largest_rating = 8
                num_5 = my_list_r.count(largest_rating)
                num_4 = my_list_r.count(second_largest_rating)

            if num_5 > 3 and num_4 > 2:
                idx_5 = [index for index, value in enumerate(my_list) if my_list_r[index] == largest_rating]
                idx_4 = [index for index, value in enumerate(my_list) if my_list_r[index] == second_largest_rating]

                select_5 = random.sample(idx_5, 3)
                select_4 = random.sample(idx_4, 2)

                select_5_i = [my_list[__] for __ in select_5]
                select_4_i = [my_list[__] for __ in select_4]

                user_like_list = []
                user_dislike_list = []
                for i_ in range(len(my_list)):
                    if my_list_r[i_] >= 4 and not (my_list[i_] in select_5 or my_list[i_] in select_4):
                        user_like_list.append(my_list[i_])
                    elif my_list_r[i_] <= 2:
                        user_dislike_list.append(my_list[i_])

                if len(user_like_list) > 55:
                    topk = 50
                user_like_list = user_like_list[:topk]
                user_dislike_list = user_dislike_list[:10]

                neglist = []
                while len(neglist) < 5:
                    rn = random.sample(movie_id_list, 1)[0]
                    if not (rn in my_list or rn in neglist):
                        neglist.append(rn)

                total_list = select_5_i + select_4_i + neglist

                total_list_mf = []
                for j_ in total_list:
                    try:
                        yy = cm_item[str(j_)]
                        uu = cm_user[str(uni)]
                        mf_label = mf_pred[uu][yy]
                    except Exception:
                        mf_label = 1.5
                    total_list_mf.append(mf_label)

                total_list_mf_idx = sort_list_reverse_with_indices(total_list_mf)
                total_list_mf_idx = total_list_mf_idx[:5]
                total_list_mf_i = [total_list[k_] for k_ in total_list_mf_idx]

                total_list_r = copy.deepcopy(total_list)
                random.shuffle(total_list_r)
                total_list_t = total_list[:5]

                historical_interactions = ', '.join([f'"{movie_name_dict[i]}"' for i in user_like_list])
                neg_interactions = ', '.join([f'"{movie_name_dict[i]}"' for i in user_dislike_list])
                true_answer_items_set = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_t])
                candidate_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_r])
                mf_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_mf_i])

                instruct0 = f'''You are a {kws} recommender system. Your task is to rank a given list of candidate {kws}s based on user preferences and return the top five recommendations.\n\n'''

                instruct1 = f'''User's Liked {kws}s: <historical_interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: How would the user rank the candidate item list: <movie_list> based to historical perference?\n'''
                instruct2 = 'Hint: Another recommender model suggests <cm_result>'
                if isHint:
                    instruct1 = instruct1 + instruct2

                instruct1 = instruct1.replace('<historical_interactions>', historical_interactions).replace(
                    '<user_unpre>', neg_interactions).replace('<movie_list>', candidate_item_sets_).replace(
                    '<cm_result>', mf_item_sets_)

                fi = {'messages': [
                    {"role": "system", "content": [{"type": "text", "content": ""}]},
                    {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1}]},
                    {"role": "assistant",
                     "content": [{"type": "text", "content": 'Answer: ' + str(true_answer_items_set)}]}
                ]}
                mes_list.append(fi)

            else:
                continue

    with jsonlines.open('/path/to/data_all.jsonl', mode='a') as writer:
        writer.write_all(mes_list)
