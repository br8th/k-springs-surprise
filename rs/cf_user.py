from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from metrics import Metrics
from evaluation_data import EvaluationData
from data.repository import RatingsRepository

def load_data():
    repository = RatingsRepository()
    print("Loading product ratings...")
    data = repository.load_customer_ratings()
    print("\nComputing product popularity ranks so we can measure novelty later...")
    rankings = repository.get_item_rankings()
    return (repository, data, rankings)

repository, data, rankings = load_data()

evalData = EvaluationData(data, rankings)

# Train on leave-One-Out train set
train_set = evalData.get_loocv_train_set()
sim_options = {
    'name': 'cosine',
    'user_based': True
}

model = KNNBasic(sim_options=sim_options)
model.fit(train_set)
# user to user similarity matrix
corr_matrix = model.compute_similarities()

# left_out_test_set = model.test(evalData.get_loocv_test_set())
left_out_test_set = evalData.get_loocv_test_set()

top_n = defaultdict(list)
k = 10

# Generate recommendations for every user in the trainset
for uiid in range(train_set.n_users):
# for uiid in train_set.all_users():
    similarity_row = corr_matrix[uiid]
    similar_users = []

    for sim_uiid, sim_score in enumerate(similarity_row):
        if sim_uiid != uiid:
            similar_users.append((sim_uiid, sim_score))

    # Get top k most similar users to this one
    k_neighbours = heapq.nlargest(k, similar_users, key= lambda t: t[1])

    # Get the stuff they rated, and add up ratings for each item, weighted by
    # user similarity
    candidates = defaultdict(float)
    for k_uid, k_sim_score in k_neighbours:
        for iid, k_rui in train_set.ur[k_uid]:
            candidates[iid] += (k_rui / 5.0) * k_sim_score

    already_rated = {}
    for iid, _ in train_set.ur[uiid]:
        already_rated[iid] = 1

    # Get top-rated items from similar users:
    n = 20
    user_id = train_set.to_raw_uid(uiid)
    
    for iid, weight in sorted(candidates.items(), key = itemgetter(1), reverse = True):
        if iid not in already_rated and n > 0:
            product_id = train_set.to_raw_iid(iid)
            top_n[user_id].append((product_id, 0.0))
            n -= 1

# Measure
print("HR", Metrics.hit_rate(top_n, left_out_test_set))
print("ARHR", Metrics.average_reciprocal_hit_rank(top_n, left_out_test_set))
