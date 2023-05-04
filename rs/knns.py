from data.repository import RatingsRepository
from surprise import KNNBasic
from surprise import NormalPredictor
from evaluator import Evaluator

import random
import numpy as np
from data.movie_lens import MovieLens


def LoadData():
    repository = RatingsRepository()
    print("Loading product ratings...")
    data = repository.load_customer_ratings()
    print("\nComputing product popularity ranks so we can measure novelty later...")
    rankings = repository.get_item_rankings()
    return (repository, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(repository, evaluation_data, rankings) = LoadData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluation_data, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
evaluator.add_algorithm(UserKNN, "User KNN")

# Item-based KNN
ItemKNN = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
evaluator.add_algorithm(ItemKNN, "Item KNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.add_algorithm(Random, "Random")

# Fight!
evaluator.evaluate(True)

evaluator.sample_top_n_recs(repository)
