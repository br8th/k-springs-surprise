from metrics import Metrics
from utils import get_top_n_predictions_per_user

class EvaluatedAlgorithm:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
    
    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm

    def evaluate(self, evaluation_data, do_top_n, n = 10, verbose = True):
        metrics = {}

        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")

        self.algorithm.fit(evaluation_data.get_train_set())
        predictions = self.algorithm.test(evaluation_data.get_test_set())

        metrics["rmse"] = Metrics.rmse(predictions)
        metrics["mae"] = Metrics.mae(predictions)

        if do_top_n:
            # Evaluate top-10 with Leave One Out testing
            if (verbose):
                print("Evaluating top-N with leave-one-out...")

            self.algorithm.fit(evaluation_data.get_loocv_train_set())
            
            # Estimate all ratings in given test set
            # left_out_predictions = self.algorithm.test(
            #     evaluation_data.get_loocv_test_set())

            left_out_predictions = evaluation_data.get_loocv_test_set()

            # Build predictions for all ratings not in the training set
            all_predictions = self.algorithm.test(
                evaluation_data.get_loocv_anti_test_set())

            # Compute top 10 recs for each user
            top_n_predicted = get_top_n_predictions_per_user(all_predictions, 20)

            if (verbose):
                print("Computing hit-rate and rank metrics...")

            # See how often we recommended a product the user actually rated
            metrics["HR"] = Metrics.hit_rate(
                top_n_predicted, left_out_predictions)

            metrics["ARHR"] = Metrics.average_reciprocal_hit_rank(
                top_n_predicted, left_out_predictions)

            # Evaluate properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")

            self.algorithm.fit(evaluation_data.get_full_train_set())
            all_predictions = self.algorithm.test(evaluation_data.get_full_anti_test_set())
            top_n_predicted = get_top_n_predictions_per_user(all_predictions, n)

            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")

            metrics["user_coverage"] = Metrics.user_coverage(
                top_n_predicted, evaluation_data.get_full_train_set().n_users, rating_threshold = 4.0)

            # Measure diversity of recommendations:
            metrics["diversity"] = Metrics.diversity(
                top_n_predicted, evaluation_data.get_full_train_set())

            # Measure novelty (average popularity rank of recommendations):
            metrics["novelty"] = Metrics.novelty(
                top_n_predicted, evaluation_data.get_popularity_rankings())

        if (verbose):
            print("Analysis complete.")

        return metrics