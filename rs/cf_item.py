from data.repository import RatingsRepository
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from metrics import Metrics
from evaluation_data import EvaluationData

repository = RatingsRepository()
print("Loading product ratings...")
data = repository.load_customer_ratings()
evalData = EvaluationData(data, False)

# Train on leave-One-Out train set
train_set = evalData.get_loocv_train_set()
sim_options = {
    'name': 'pearson',
    'user_based': False
}

model = KNNBasic(sim_options = sim_options)
model.fit(train_set)
# item to item similarity matrix
corr_matrix = model.compute_similarities()

# test = evalData.get_loocv_test_set()
# left_out_test_set = model.test(test)
left_out_test_set = evalData.get_loocv_test_set()

# Build up dict to lists of (productID, predicted_rating) pairs
# 15 => (346, 0.0), (122, 0.0)
# 84 => (412, 0.0)
top_n = defaultdict(list)
k = 10


print('train_set : num_users: {} num_items: {}...'.format(train_set.n_users, train_set.n_items))
# print('test_set  : num_users: {} num_items: {}...'.format(len(left_out_test_set), 1))

# Generate recommendations for every single user in the trainset
for uiid in range(train_set.n_users):
# for uiid in train_set.all_users():

    user_id = train_set.to_raw_uid(uiid)
    n = 20
    # print('Generating top {} recommendations for user {}...'.format(n, user_id))

    user_ratings = train_set.ur[uiid]
    k_user_ratings = [(iid, r_ui ) for iid, r_ui in user_ratings if r_ui >= 4.0 ]

    # Get top k items rated by this user, sorted by rating
    k_neighbours = heapq.nlargest(k, k_user_ratings, key = lambda t: t[1])
    candidates = defaultdict(float)

    for k_iid, k_rui in k_neighbours:
        # 2. get items similar to current item
        similarity_row = corr_matrix[k_iid]
        for iid, sim_score in enumerate(similarity_row):
            candidates[iid] += (k_rui / 5.0) * sim_score

    # 3. generate candidates and filter out
    already_rated = {}
    for iid, _ in user_ratings:
        already_rated[iid] = 1
    
    results = []

    for c_iid, c_weight in sorted(candidates.items(), key = itemgetter(1), reverse = True):
        if c_iid not in already_rated and n > 0:
            product_id = train_set.to_raw_iid(c_iid)
            print(repository.get_product_name(product_id), c_weight)
            results.append(product_id)
            top_n[user_id].append((product_id, 0.0))
            n -= 1
    
    # print('###############################\n')

# Measure
print("HR", Metrics.hit_rate(top_n, left_out_test_set))
print("ARHR", Metrics.average_reciprocal_hit_rank(top_n, left_out_test_set))

class ItemCF:

    def __init__(self, eval_data):
        self.eval_data = eval_data
        return

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
        for iid, rating_sum in sorted(candidates.items(), key = itemgetter(1), reverse = True):
            if iid not in already_rated:
                riid = train_set.to_raw_iid(iid)
                print(repository.get_product_name(riid), rating_sum)
                product_ids.append(riid)
                n -= 1
                if (n == 0):
                    break

        return repository.get_products_details(product_ids)

algo = ItemCF(36)

# print(algo.get_top_n_for_users())