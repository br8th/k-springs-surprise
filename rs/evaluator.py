from evaluation_data import EvaluationData
from evaluated_algorithm import EvaluatedAlgorithm


class Evaluator:

    algorithms = []

    def __init__(self, dataset, rankings):
        self.dataset = EvaluationData(dataset, rankings)

    def add_algorithm(self, algorithm, name):
        self.algorithms.append(EvaluatedAlgorithm(algorithm, name))

    def evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            name = algorithm.get_name()
            print("Evaluating ", name, "...")
            results[name] = algorithm.evaluate(self.dataset, doTopN)

        # Print results
        print("\n# Results\n")

        if (doTopN):
            print(
                "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algo",
                    "RMSE",
                    "MAE",
                    "HR",
                    "ARHR",
                    "Coverage",
                    "Diversity",
                    "Novelty"))
            for (name, metrics) in results.items():
                print(
                    "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name,
                        metrics["rmse"],
                        metrics["mae"],
                        metrics["HR"],
                        metrics["ARHR"],
                        metrics["user_coverage"],
                        metrics["diversity"],
                        metrics["novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "rmse", "mae"))
            for (name, metrics) in results.items():
                print(
                    "{:<10} {:<10.4f} {:<10.4f}".format(
                        name,
                        metrics["rmse"],
                        metrics["mae"]))

        print("\n# Legend\n")
        print("RMSE:      Root Mean Squared Error. A lower score means better accuracy.")
        print("MAE:       Mean Absolute Error.A lower values mean better accuracy.")

        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
            print("Coverage:  Proportion of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommended items")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    def sample_top_n_recs(self, repository, uid = 15, k = 10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.get_name())
            print("\nBuilding recommendation model...")

            train_set = self.dataset.get_full_train_set()
            algo.get_algorithm().fit(train_set)

            print("Computing recommendations...")
            test_set = self.dataset.get_anti_test_set_for_user(uid)

            predictions = algo.get_algorithm().test(test_set)

            recommendations = []

            print("\nWe recommend:")
            for uid, iid, r_ui, pr_ui, _ in predictions:
                recommendations.append((iid, pr_ui))

            recommendations.sort(key = lambda x: x[1], reverse = True)

            for ratings in recommendations[:10]:
                print(repository.get_product_name(ratings[0]), ratings[1])
