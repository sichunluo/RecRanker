import random
like = []
dislike = []
with open('ratings.csv') as f:
    for line in f:
        items = line.strip().split('\t')
        new_line = ','.join(items[:])+'\n'
        if int(items[-2])<4:
            if int(items[-2])<3:
                dislike.append(new_line)
            continue
        like.append(new_line)

with open('like.txt','w') as f:
    f.writelines(like)

with open('dislike.txt','w') as f:
    f.writelines(dislike)
import pandas as pd
df_like = pd.read_csv('like.txt',header=None, names=['u', 'i', 'r','t'])
sorted_data = df_like.sort_values(by=['u', 't'])
train_set = pd.DataFrame(columns=['u'])
val_set = pd.DataFrame(columns=['u'])
test_set = pd.DataFrame(columns=['u'])
for user_id, group in sorted_data.groupby('u'):
    num_ratings = len(group)
    test_set = pd.concat([test_set, group.iloc[[num_ratings-1]]])
    val_set = pd.concat([val_set, group.iloc[[num_ratings-2]]])
    train_set = pd.concat([train_set, group.iloc[:num_ratings-2]])
train_set.reset_index(drop=True, inplace=True)
val_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)

train_set=train_set.astype(int)
val_set=val_set.astype(int)
test_set=test_set.astype(int)
train_set.to_csv('train_set.txt', index=False, header=None, sep=' ')
val_set.to_csv('valid_set.txt', index=False, header=None, sep=' ')
test_set.to_csv('test_set.txt', index=False, header=None, sep=' ')
print("训练集（Training set）:")
print(train_set)
print("\n验证集（Validation set）:")
print(val_set)
print("\n测试集（Test set）:")
print(test_set)

