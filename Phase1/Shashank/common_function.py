from collections import defaultdict

from math import sqrt


def fetch_data(file_name):
    """
    :return: a counter for user, this will be a vector kind of concept
    """
    d = {}
    for user in open('../{0}/{1}'.format('Data', file_name)).readlines():
        user = user.split(' ')
        user_id = user.pop(0)
        d[user_id] = defaultdict(lambda: {})
        while len(user) > 1:
            key, tf, df, tfidf = user.pop(0), float(user.pop(0)), float(user.pop(0)), float(user.pop(0))
            d[user_id][key] = {'TF': tf, 'DF': df, 'TF-IDF': tfidf}
    return d


def cosine_similarity(u1, u2, model):
    intersect = set(u1.keys()) & set(u2.keys())
    numerator, denominator = sum([u1[x][model] * u2[x][model] for x in intersect]), sqrt(
        sum([u1[x][model] ** 2 for x in u1])) * sqrt(sum([u2[x][model] ** 2 for x in u2]))
    if not denominator:
        return 0.0, []
    else:
        return float(numerator) / float(denominator), sorted([x for x in intersect],
                                                             key=lambda x: (-u1[x][model] * u2[x][model], x))[:3]
