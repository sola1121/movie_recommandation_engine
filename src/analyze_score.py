import numpy as np


def euclidean_score(dataset, user1, user2):
    """dataset: 用户, 电影, 评分的数据集, user1: 用户1, user2: 用户2
    计算user1和user2用户之间的欧几里得距离
    """
    if user1 not in dataset:
        raise TypeError("User " + user1 + " not present in dataset.")
    if user2 not in dataset:
        raise TypeError("User " + user2 + " not present in dataset.")

    # 提取两个用户均评过分的电影
    rated_by_both = dict()

    for item in dataset[user1]:        # 获取用户1的电影字典的keys, 即电影名
        if item in dataset[user2]:     # 如果该电影名在用户2的电影字典的keys中
            rated_by_both[item] = 1    # 将电影名作为键, 得分为1
    
    # 如果两个用户没有相同的电影, 不能说这两个用户行为间有关联, 这里设置得分为0
    if len(rated_by_both) == 0:
        return 0

    # 对于每个共同评分, 只计算平方和的平方根, 并将值归一化
    squared_differences = list()

    for item in dataset[user1]:        # 在用户1中的电影字典keys, 即电影名
        if item in dataset[user2]:     # 在用户2中有相同的电影
            squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))   # 用户的评分的差的平方
    
    return 1 / (1 + np.sqrt(np.sum(squared_differences)))   # 求和后开平方并归一化


def pearson_score(dataset, user1, user2):
    """dataset: 用户, 电影, 评分的数据集, user1: 用户1, user2: 用户2
    计算user1和user2的皮尔逊积矩相关系数
    """
    if user1 not in dataset:
        raise TypeError("User " + user1 + " not present in dataset.")
    if user2 not in dataset:
        raise TypeError("User " + user2 + " not present in dataset.")

    # 提取两个用户均评过分的电影
    rated_by_both = dict()

    for item in dataset[user1]:         # 获取用户1的电影字典的keys, 即电影名
        if item in dataset[user2]:      # 如果该电影名在用户2的电影字典的keys中
            rated_by_both[item] = 1     # 将电影名作为键, 得分为1
    
    num_ratings = len(rated_by_both)
    # 如果两个用户都没有评分, 得分为0
    if num_ratings == 0:
        return 0

    # 两个用户都评过分的电影的评分之和
    user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])
    user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

    # 两个用户都评过分的电影的评分的平方和
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

    # 计算数据集的乘积之和
    product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])

    # 计算皮尔逊积矩相关系数
    Sxy =product_sum - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    # 考虑分母为0的情况
    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)
