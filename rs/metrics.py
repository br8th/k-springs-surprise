import itertools

from surprise import accuracy
from surprise import KNNBaseline

class Metrics:

    @staticmethod
    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)

    @staticmethod
    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)

    @staticmethod
    def hit_rate(top_n_predicted, test_set_predictions):
        """
            A hit is the number of items in the test set that were also
            present in the top-N items recommended to a user.

        Parameters
        ----------
        top_n_predicted : dict
            The top N predictions for all the users in the test set
        test_set_predictions : list 
            Predictions (uid, iid, r_ui, pr_ui) made on the test set.
            The test set was excluded from the training data

        Returns
        -------
            hit-rate (HR) = num_hits / num_users
            A HR of 1.0 indicates that the algo was always able to recommend
            any of the hidden items
        """
        hits = 0

        # for test_uid, test_iid, r_ui, pr_ui, _ in test_set_predictions:
        for test_uid, test_iid, r_ui in test_set_predictions:
            for iid, _ in top_n_predicted[test_uid]:
                if test_iid == iid:
                    hits += 1
                    break

        # Compute overall precision
        return hits / len(test_set_predictions)
    
    @staticmethod
    def average_reciprocal_hit_rank(top_n_predicted, test_set_predictions):
        summation = 0

        # for test_uid, test_iid, r_ui, pr_ui, _ in test_set_predictions:
        for test_uid, test_iid, r_ui in test_set_predictions:
            hit_rank = 0
            rank = 0
            for iid, pr_ui in top_n_predicted[test_uid]:
                rank = rank + 1
                if test_iid == iid:
                    hit_rank = rank
                    break
            if hit_rank > 0:
                summation += 1.0 / hit_rank

        return summation / len(test_set_predictions)

    # The proportion of users for which the system can predict 'good' ratings.
    @staticmethod
    def user_coverage(top_n_predicted, num_users, rating_threshold = 4.0):
        hits = 0

        for uid, predicted_ratings in top_n_predicted.items():
            hit = False
            for iid, pr_ui in predicted_ratings:
                if pr_ui >= rating_threshold:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / num_users
    
    @staticmethod
    def diversity(top_n_predicted, full_train_set):
        
        # item-item similarity is used to measure diversity.
        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        diversity_algo = KNNBaseline(sim_options = sim_options)
        diversity_algo.fit(full_train_set)

        n = 0
        total = 0
        sims_matrix = diversity_algo.compute_similarities()

        for uid, predicted_ratings in top_n_predicted.items():
            pairs = itertools.combinations(predicted_ratings, 2)
            for pair1, pair2 in pairs:
                inner_id_1 = diversity_algo.trainset.to_inner_iid(pair1[0])
                inner_id_2 = diversity_algo.trainset.to_inner_iid(pair2[0])
                similarity = sims_matrix[inner_id_1][inner_id_2]
                total += similarity
                n += 1

        S = total / n
        return (1 - S)

    # We assume that more popular items are less novel
    @staticmethod
    def novelty(user_predictions, popularity):
        n = 0
        total = 0

        for uid, predicted_ratings in user_predictions.items():
            n += len(predicted_ratings)
            for iid, _ in predicted_ratings:
                total += popularity[iid]
        
        return total / n
