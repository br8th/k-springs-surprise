from data.repository import RatingsRepository
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

class Recommender:

    @staticmethod
    def get_top_n_for_user(n = 10, k = 10, uid = 15):
        repository = RatingsRepository()
        data = repository.load_customer_ratings()
        train_set = data.build_full_trainset()

        model = KNNBasic(
            sim_options = {
                'name': 'pearson',
                'user_based': False
            }
        )

        model.fit(train_set)
        corr_matrix = model.compute_similarities()
        print(corr_matrix)

        iuid = train_set.to_inner_uid(uid)

        # Get the top K items we rated
        my_ratings = train_set.ur[iuid]
        k_neighbors = heapq.nlargest(k, my_ratings, key = lambda t: t[1])

        # Get similar items to stuff we liked (weighted by rating)
        candidates = defaultdict(float)
        for k_iid, k_r_ui in k_neighbors:
            my_corr_row = corr_matrix[k_iid]
            for iid, score in enumerate(my_corr_row):
                candidates[iid] += score * (k_r_ui / 5.0)

        # Items the user has already rated
        already_rated = {}
        for iid, r_ui in train_set.ur[iuid]:
            already_rated[iid] = 1

        product_ids = []

        # Get top-rated items from similar users:
        for iid, ratingSum in sorted(candidates.items(), key = itemgetter(1), reverse = True):
            if iid not in already_rated:
                riid = train_set.to_raw_iid(iid)
                print(repository.get_product_name(riid), ratingSum)
                product_ids.append(riid)
                n -= 1
                if (n == 0):
                    break

        return repository.get_products_details(product_ids)