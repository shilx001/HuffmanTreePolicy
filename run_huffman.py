import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from huffman import *
import datetime
import matplotlib.pyplot as plt
from funk_svd import SVD

model_name = 'GRU'
np.random.seed(1)
tf.set_random_seed(1)
#data = pd.read_csv('ratings.csv', header=0, names=['u_id', 'i_id', 'rating', 'timestep'])
# data = pd.read_table('ratings.dat', sep='::', names=['u_id', 'i_id', 'rating', 'timestep'])
# data=data[:100]
data = pd.read_csv('amazon_Baby.csv', header=0, names=['index', 'u_id', 'i_id', 'rating', 'timestep'])
del data['index']
data = data[:len(data)//4]

user_idx = data['u_id'].unique()  # id for all the user
np.random.shuffle(user_idx)
user_idx = user_idx[:len(user_idx)]
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# Count the movies
movie_id = []
'''
for idx1 in user_idx:  # 针对train_id中的每个用户
    user_record = data[data['u_id'] == idx1]
    for idx2, row in user_record.iterrows():
        if row['i_id'] in movie_id:
            idx = movie_id.index(row['i_id'])  # 找到该位置
        else:
            # 否则新加入movie_id
            movie_id.append(row['i_id'])
'''
# count the movies
movie_total_rating = []
movie_count = []
movie_id = []
for idx1 in user_idx:  # 针对train_id中的每个用户
    user_record = data[data['u_id'] == idx1]
    for idx2, row in user_record.iterrows():
        # 检查list中是否有该电影的id
        if row['i_id'] in movie_id:
            idx = movie_id.index(row['i_id'])  # 找到该位置
            movie_count[idx] += 1  # 计数加一
            movie_total_rating[idx] += row['rating']
        else:
            # 否则新加入movie_id
            movie_id.append(row['i_id'])
            movie_count.append(1)
            movie_total_rating.append(row['rating'])
print('Total movie count:', len(movie_id))

# Funk SVD for item representation
train = data[data['u_id'].isin(train_id)]
test = data[data['u_id'].isin(test_id)]
svd = SVD(learning_rate=1e-3, regularization=0.005, n_epochs=200, n_factors=128, min_rating=0, max_rating=5)
svd.fit(X=data, X_val=test, early_stopping=True, shuffle=False)
item_matrix = svd.qi
user_matrix = svd.pu


def get_feature(input_id):
    # 根据输入的movie_id得出相应的feature
    movie_index = np.where(movie_id == input_id)
    return item_matrix[movie_index]

def get_user_feature(input_id):
    # 根据用户id找到相应feature
    return user_matrix[input_id - 1]

def action_mapping(input_id):
    '''
    convert input movie id to index
    :param input_id: movie id
    :return: index of movie.
    '''
    return np.where(movie_id == input_id)


def one_hot_rating(input_rating):
    '''
    convert the rating to one-hot code.
    :param input_rating:
    :return: one-hot code for ratings
    '''
    output_rating = np.zeros(11)
    index = int(input_rating / 0.5)
    output_rating[index] = 1
    return output_rating


max_seq_length = 32
state_dim = item_matrix.shape[1] + 11
feature_dim = item_matrix.shape[1]
hidden_size = 16
branch = item_matrix.shape[0]

agent = HuffmanTree(value=movie_count, id=list(range(len(movie_count))), state_dim=state_dim,
                    user_state_dim=item_matrix.shape[1], branch=branch, hidden=hidden_size,
                    learning_rate=1e-3, max_seq_length=max_seq_length)


def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


print('Begin training the tree policy.')
start = datetime.datetime.now()
discount_factor = 1
train_step = 0
loss_list = []
for id1 in train_id:
    user_record = data[data['u_id'] == id1]
    #user_record = user_record.sample(frac=1)
    user_record = user_record.sort_values(by='timestep')
    state = []
    rating = []
    action = []
    cnt = 0
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['i_id'])
        current_state = np.hstack((movie_feature.flatten(), one_hot_rating(row['rating'])))
        rating.append((row['rating']))
        state.append(current_state)
        action.append(action_mapping(row['i_id']))
        cnt += 1
        if cnt % max_seq_length == 0 or (cnt % max_seq_length > 1 and cnt >= len(user_record)):
            state_list = []
            state_length_list = []
            action_list = []
            reward_list = []
            for i in range(1, len(state)):
                current_state = state[:i]
                current_state_length = i
                temp_state = agent.state_padding(current_state, current_state_length)
                state_length_list.append(current_state_length)
                state_list.append(temp_state)
                action_list.append(action[i])
                reward_list.append(normalize(rating[i]))  # normalization of the ratings to 0,1
            discount = discount_factor ** np.arange(len(reward_list))
            Q_value = reward_list * discount
            Q_value = np.cumsum(Q_value[::-1])[::-1]
            user_feature = get_user_feature(id1)
            input_user_feature = np.reshape(user_feature, [-1, feature_dim]).repeat(len(state_list), axis=1)
            loss = agent.learn(input_user_feature, state_list, np.array(state_length_list), np.array(action_list), Q_value)
            loss_list.append(loss)
            train_step += 1
            print('Step ', train_step, 'Loss: ', loss)
            state = []
            rating = []
            action = []
end = datetime.datetime.now()
training_time = (end - start).seconds

print('Begin Test')
test_count = 0
result = []
total_testing_steps = 0
start = datetime.datetime.now()


def evaluate(recommend_id, item_id, rating, top_N):
    '''
    evalute the recommend result for each user.
    :param recommend_id: the recommend_result for each item, a list that contains the results for each item.
    :param item_id: item id.
    :param rating: user's rating on item.
    :param top_N: N, a real number of N for evaluation.
    :return: reward@N, recall@N, MRR@N
    '''
    session_length = len(recommend_id)
    relevant = 0
    recommend_relevant = 0
    selected = 0
    output_reward = 0
    mrr = 0
    if session_length == 0:
        return 0, 0, 0, 0
    for ti in range(session_length):
        current_recommend_id = list(recommend_id[ti])[:top_N]
        current_item = item_id[ti]
        current_rating = rating[ti]
        if current_rating > 3.5:
            relevant += 1
            if current_item in current_recommend_id:
                recommend_relevant += 1
        if current_item in current_recommend_id:
            selected += 1
            output_reward += normalize(current_rating)
            rank = current_recommend_id.index(current_item)
            mrr += 1.0 / (rank + 1)
    recall = recommend_relevant / relevant if relevant != 0 else 0
    precision = recommend_relevant / session_length
    return output_reward / session_length, precision, recall, mrr / session_length


result_analysis = []

for id1 in test_id:
    user_record = data[data['u_id'] == id1]
    #user_record = user_record.sample(frac=1)
    user_record = user_record.sort_values(by='timestep')
    all_state = []
    all_recommend = []
    all_item = []
    all_rating = []
    test_count += 1
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['i_id'])
        current_state = np.hstack((movie_feature.flatten(), one_hot_rating(row['rating'])))
        all_state.append(current_state)
        if len(all_state) > 1:
            temp_state = all_state[:-1]
            temp_state_length = len(temp_state)
            temp_state = agent.state_padding(temp_state, temp_state_length)
            output_action = agent.get_all_action_prob(get_user_feature(id1),temp_state, temp_state_length).flatten()
            total_testing_steps += 1
            output_action = output_action[:len(movie_id)]
            recommend_idx = np.argsort(-output_action)[:100]
            recommend_movie = [movie_id[_] for _ in recommend_idx]
            all_recommend.append(recommend_movie)
            all_item.append(row['i_id'])
            all_rating.append(row['rating'])
    if len(all_rating) > 0:
        reward_10, precision_10, recall_10, mkk_10 = evaluate(all_recommend, all_item, all_rating, 10)
        reward_30, precision_30, recall_30, mkk_30 = evaluate(all_recommend, all_item, all_rating, 30)
        result_analysis.append((all_recommend, all_item, all_rating))
        print('Test user #', test_count, '/', len(test_id))
        print('Reward@10: %.4f, Precision@10: %.4f, Recall@10: %.4f, MRR@10: %4f'
            % (reward_10, precision_10, recall_10, mkk_10))
        print('Reward@30: %.4f, Precision@30: %.4f, Recall@30: %.4f, MRR@30: %4f'
            % (reward_30, precision_30, recall_30, mkk_30))
        result.append([reward_10, precision_10, recall_10, mkk_10, reward_30, precision_30, recall_30, mkk_30])
end = datetime.datetime.now()
testing_time = (end - start).seconds

print('###############')
print('Learning finished')
print('Total training steps: {}'.format(train_step))
print('Total learning time: {}'.format(training_time))
print('Average learning time for each step: {:.5f}'.format(training_time / train_step))
print('Total testing steps: {}'.format(total_testing_steps))
print('Total testing time: {}'.format(testing_time))
print('Average time per decision: {:.5f}'.format(testing_time / total_testing_steps))

#pickle.dump(result, open('huffman' + model_name, mode='wb'))
print('Result:')
display = np.mean(np.array(result).reshape([-1, 8]), axis=0)
for num in display:
    print('%.5f' % num)
#pickle.dump(result_analysis, open('huffman_analysis', mode='wb'))
