from analyze_score import np, euclidean_score, pearson_score


def find_similar_users(dataset, user, num_users, kernel="pearson"):
    """dataset: 数据集, user: 查找的用户, num_users: 想要找到的相似用户个数
    第一步将会查看该用户是否包含在数据集中.
    第二步如果用户存在, 则需要计算该用户与数据集中其他所有用户的皮尔逊积矩相关系数
    """
    if user not in dataset:
        raise TypeError("User" + user + " not present in the dataset.")
    if kernel not in ("pearson", "euclidean"):
        raise AttributeError("Kernel " + kernel + " not suport.")

    scores = None
    # 根据使用的核心调用不同的得分计算方式, 生成所有其他用户和当前用户的关联得分
    # 生成 [[其他用户1名字, 与当前用户关联得分], [其他用户2名字, 与当前用户关联得分], ...] 的列表
    if kernel == "pearson":
        scores = np.array([[name, pearson_score(dataset, user, name)] for name in dataset if user != name])
    elif kernel == "euclidean":
        scores = np.array([[name, euclidean_score(dataset, user, name)] for name in dataset if user != name])

    # 评分按照第二列排列
    scores_sorted = np.argsort(scores[:, 1])

    # 评分按照降序排列
    scores_sorted_dec = scores_sorted[::-1]

    # 提取出k个最高分
    top_k = scores_sorted_dec[0:num_users]

    return scores[top_k]


def generate_recommendations(dataset, user, kernel="pearson"):
    """dataset: 用户, 电影, 评分的数据集, user: 被在数据集中评估的用户
    生成针对user的电影推荐, 并根据相关性排列
    """
    if user not in dataset:
        raise TypeError("User " + user + " not present in dataset")
    if kernel not in ("pearson", "euclidean"):
        raise AttributeError("Kernel " + kernel + " not suport.")

    score_caculator = None
    if kernel == "pearson":
        score_caculator = pearson_score
    elif kernel == "euclidean":
        score_caculator = euclidean_score

    # 计算该用户与数据集中其他用户的皮尔逊相关系数
    total_scores = dict()   
    similarity_sums = dict()

    for other_user in [u for u in dataset if u != user]:
        similarity_score = score_caculator(dataset, user, other_user)   # 当前用户和另一个用户的关联得分

        if similarity_score <= 0:
            continue

        # 找到未被该用户评分的电影
        for item in [m for m in dataset[other_user] if m not in dataset[user] or dataset[user][m] == 0]:
            total_scores.update({item: dataset[other_user][item] * similarity_score})   # 更新或添加当前电影的另一个用户的评分*关联得分
            similarity_sums.update({item: similarity_score})   # 更新或添加当前电影的关联得分

    # 如果该用户看过数据集中所有的电影, 那就不能为用户推荐电影
    if len(total_scores) == 0:
        return ['No recommendations possible']

    # 生成一个电影评分标准化列表
    movie_ranks = np.array([[total/similarity_sums[item], item]
                            for item, total in total_scores.items()])

    # 根据第一列对皮尔逊相关系数进行降序排列
    movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

    # 提取出推荐电影
    recommendations = [movie for _, movie in movie_ranks]

    return recommendations


if __name__ == "__main__":

    import os
    import json


    file_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    file = open(file_dir + "/movie_ratings.json", 'r')
    data = json.loads(file.read())
    file.close()

    # 选取用户进行测试
    user = "William Reynolds"   # Michael Henry"
    print("\nRecommandations for " + user + ":")
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(i+1, " -- ", movie)
    
    # 看过所有数据集中的电影的用户
    user = "John Carson"
    print("\nRecommandations for " + user + ":")
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(i+1, " -- ", movie)
    