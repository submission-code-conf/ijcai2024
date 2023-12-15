from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
import faiss
# 定义一个函数来计算一个用户对所有电影的评分预测值
def predict_all_scores(user_id, user_emb_dict, movie_emb_dict, movie_ids):
    # 创建一个空列表，用来存储预测值
    scores = []
    # 获取用户的embedding
    user_embedding = user_emb_dict[user_id]
    # 遍历所有电影
    for movie_id in movie_ids:
        # 获取电影的embedding
        movie_embedding = movie_emb_dict[movie_id]
        # https://zhuanlan.zhihu.com/p/508625294
        # sklearn内置的cosine_similarity特别慢
        score = user_embedding.dot(movie_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(movie_embedding))
        score = score.tolist()
        # 将预测值添加到列表中
        scores.append(score)
    # 返回预测值列表
    return np.array(scores)

# 定义一个函数来根据预测值列表对所有电影进行排序
def rank_movies(scores):
    # 使用numpy.argsort()方法，得到一个按照预测值降序排列的电影索引列表
    ranked_movies = np.argsort(scores)[::-1]
    # 返回排序后的电影索引列表
    return ranked_movies


# 定义一个函数来截取排序后的电影索引列表的前K个元素，作为推荐列表
def get_top_k_movies(ranked_movies, k):
    # 使用列表切片操作，得到一个长度为K的推荐列表
    top_k_movies = ranked_movies[:k]
    # 返回推荐列表
    return top_k_movies


# 定义一个函数来判断推荐列表中是否包含用户真正感兴趣的电影
def is_hit(top_k_movies, user_id, user_movie_dict):
    # 使用user_movies数据作为真实标签，获取用户感兴趣的电影列表
    true_movies = user_movie_dict[user_id]
    # 判断推荐列表中是否有与真实标签相同的电影，如果有则返回True，否则返回False
    for movie in top_k_movies:
        if movie in true_movies:
            return True
    return False


# 定义一个函数来计算推荐列表中包含用户真正感兴趣的电影的个数
def get_hit_count(top_k_movies, user_id, user_movie_dict):
    # 使用user_movies数据作为真实标签，获取用户感兴趣的电影列表
    true_movies = user_movie_dict[user_id]
    # 创建一个变量，用来存储命中个数
    hit_count = 0
    hit_position = []
    # print("top_k_movies: ", top_k_movies)
    # print("true_movies: ", true_movies)
    # 遍历推荐列表中的每个电影，如果与真实标签相同，则命中个数加一
    for i, movie in enumerate(top_k_movies):
        if movie in true_movies:
            hit_count += 1
            hit_position.append(i + 1)
    # 返回命中个数, 返回命中位置列表
    return hit_count, hit_position


# 定义一个函数来计算HR和NDCG指标
def evaluate_rec_scores(k, user_movie_dict, user_emb_dict, movie_emb_dict):
    # 创建两个变量，用来存储HR和NDCG的累加值
    hr_sum = 0
    ndcg_sum = 0
    # 获取用户数
    user_count = 0
    # 获取全部的movie_ids
    movie_ids = []
    movie_ids_emb = []
    for movie_id in movie_emb_dict.keys():
        movie_ids.append(movie_id)
        movie_ids_emb.append(movie_emb_dict[movie_id] / np.linalg.norm(movie_emb_dict[movie_id]))
    movie_ids_array = np.array(movie_ids)
    index = faiss.IndexFlatL2(64)
    movie_ids_emb = np.array(movie_ids_emb).astype('float32')
    # print(movie_ids_emb)
    index.add(movie_ids_emb)
    # print(index.ntotal)  # 加入了多少行数据

    # 遍历所有用户
    for user_id in user_emb_dict.keys():
        user_movie = user_movie_dict[user_id]
        if len(user_movie) == 0:
            continue
        user_count += 1
        if user_count % 100 == 0:
            print("user_count:", user_count)
        # 调用上述函数，得到预测值列表，排序后的电影索引列表，和推荐列表
        # scores = predict_all_scores(user_id, user_emb_dict, movie_emb_dict, movie_ids)
        # ranked_movies = rank_movies(scores)
        # 返回的是索引，要找到对应的movie id
        # top_k_movies_index = get_top_k_movies(ranked_movies, k)
        user_embedding = user_emb_dict[user_id]
        search_emb = user_embedding / np.linalg.norm(user_embedding)
        # print(search_emb.shape)
        search_emb = search_emb.reshape((1, 64)).astype('float32')
        # D和I分别表示筛选出来的dis和index
        D, I = index.search(search_emb, k)
        I = I.reshape((k))
        top_k_movies = movie_ids_array[I]
        # 调用上述函数，判断是否有命中，获取命中个数和位置
        hit_count, hit_position = get_hit_count(top_k_movies, user_id, user_movie_dict)
        # 如果有命中，则计算HR和NDCG，并累加到对应的变量中
        hr_sum += min(hit_count, 1)
        dcg = 0
        idcg = 0
        # 计算DCG，根据公式，对每个命中位置应用对数折扣，并累加
        for pos in hit_position:
            dcg += 1 / np.log2(pos + 1)
        # 计算IDCG，根据公式，对每个命中个数应用对数折扣，并累加
        for i in range(hit_count):
            idcg += 1 / np.log2(i + 2)
        # 计算NDCG，根据公式，将DCG除以IDCG
        ndcg = dcg / (idcg + 1e-6)
        ndcg_sum += ndcg
        # print(id, user_id, hr, ndcg, len(user_movie_dict[user_id]))

    # 计算平均HR和NDCG，根据公式，将累加值除以用户数
    hr_avg = hr_sum / user_count
    ndcg_avg = ndcg_sum / user_count
    # 返回平均HR和NDCG
    print("user_count: ", user_count)
    print("hr_avg: ", hr_avg)
    print("ndcg_avg: ", ndcg_avg)
    return hr_avg, ndcg_avg

# 分别读dataframe
ratings = pd.read_csv("tzzs_data1.csv")
user_item_emb = pd.read_csv("tzzs_data2.csv")

# 读取对应的rating
user_ids = ratings["user_id"].values
movie_ids = ratings["movie_id"].values
ratings = ratings["user_rating"].values

print("rating before: ", ratings)
ratings = np.where(ratings >= 3, 1.0, 0.0)
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
evaluate_rec_scores(50, user_movie_dict, user_emb_dict, movie_emb_dict)