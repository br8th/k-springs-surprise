from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut


class EvaluationData:

    def __init__(self, data, popularity_rankings):

        self.rankings = popularity_rankings

        # Build a full training set for evaluating overall properties
        self.full_train_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_train_set.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.train_set, self.test_set = train_test_split(
            data, test_size = .25, random_state = 1)

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        loocv = LeaveOneOut(n_splits = 1, random_state = 1, min_n_ratings = 1)
        for train, test in loocv.split(data):
            self.loocv_train = train
            self.loocv_test = test

        self.loocv_anti_test_set = self.loocv_train.build_anti_testset()

    def get_full_train_set(self):
        return self.full_train_set

    def get_full_anti_test_set(self):
        return self.full_anti_test_set

    def get_anti_test_set_for_user(self, uid):
        trainset = self.full_train_set
        fill = trainset.global_mean
        iuid = trainset.to_inner_uid(uid)
        user_items = set([j for (j, _) in trainset.ur[iuid]])
        anti_testset = []
        anti_testset += [(uid, trainset.to_raw_iid(i), fill)
                         for i in trainset.all_items() if i not in user_items]
        return anti_testset

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

    def get_loocv_train_set(self):
        return self.loocv_train

    def get_loocv_test_set(self):
        return self.loocv_test

    def get_loocv_anti_test_set(self):
        return self.loocv_anti_test_set

    def get_popularity_rankings(self):
        return self.rankings
