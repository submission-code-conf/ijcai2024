from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re

# 分别读dataframe
ratings = pd.read_csv("tzzs_data1_bc.csv")
user_item_emb = pd.read_csv("tzzs_data2_bc.csv")

# 读取对应的rating
user_ids = ratings["User-ID"].values
movie_ids = ratings["ISBN"].values
ratings = ratings["Book-Rating"].values

print("rating before: ", ratings)
ratings = np.where(ratings >= 8, 1.0, 0.0)
print(ratings)

def get_np(key, df):
    values = df[key].values
    # print(key, values)
    result = []
    for item in values:
        #将多个连续的空格合并成一个
        item = re.sub(r"\s+", " ", item)
        item = item.replace("\n", "").replace("[", "").replace("]", "").strip().replace(" ", ",")
        numpy_array = np.fromstring(item, dtype=float, sep=',')
        result.append(numpy_array)

    return np.array(result)

user_emb_atom = get_np("query_atom_repr", user_item_emb)
user_emb_side = get_np("query_side_repr", user_item_emb)
user_embs = np.concatenate((user_emb_atom, user_emb_side), axis=-1)
print("user_emb_atom: ", user_emb_atom.shape)
print("user_emb_side: ", user_emb_side.shape)
print("user_emb: ", user_embs.shape)

item_emb_atom = get_np("item_atom_repr", user_item_emb)
item_emb_side = get_np("item_side_repr", user_item_emb)
item_embs = np.concatenate((item_emb_atom, item_emb_side), axis=-1)
print("item_emb_atom: ", item_emb_atom.shape)
print("item_emb_side: ", item_emb_side.shape)
print("item_emb: ", item_embs.shape)

# 分别是每个用户点了哪些movie，user对应的emb，item对应的emb
user_movie_dict, user_emb_dict, movie_emb_dict = {}, {}, {}
user_ids = user_ids.tolist()
movie_ids = movie_ids.tolist()
ratings = ratings.tolist()
user_embs = user_embs.tolist()
item_embs = item_embs.tolist()
print(len(user_ids), len(movie_ids), len(ratings), len(user_embs), len(item_embs))

for user_id, movie_id, rating, user_emb, item_emb in zip(user_ids, movie_ids, ratings, user_embs, item_embs):
    user_emb = np.array(user_emb)
    item_emb = np.array(item_emb)
    if user_id in user_emb_dict.keys():
        assert np.max(np.abs(user_emb - user_emb_dict[user_id])) <= 0.01
    else:
        user_emb_dict[user_id] = user_emb
    if movie_id in movie_emb_dict.keys():
        assert np.max(np.abs(item_emb - movie_emb_dict[movie_id])) <= 0.01
    else:
        movie_emb_dict[movie_id] = item_emb
    if user_id not in user_movie_dict.keys():
        user_movie_dict[user_id] = []
    if rating >= 1.0:
        user_movie_dict[user_id].append(movie_id)

print("user_movie_dict: ", len(user_movie_dict.keys()))
print("user_emb_dict: ", len(user_emb_dict.keys()))
print("movie_emb_dict: ", len(movie_emb_dict.keys()))

count = 0
for value in user_movie_dict.values():
    if len(value) > 0:
        count += 1

print("count:", count)