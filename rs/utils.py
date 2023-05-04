from collections import defaultdict

"""The utils module contains the get_top_n_predictions_per_user function."""

@staticmethod
def get_top_n_predictions_per_user(all_predictions, n = 10, minimum_rating = 3.7):
    """Gets the top N prediction for each user in a given set of predictions.

    Parameters
    ----------
    all_predictions : list of Predictions (uid, iid, r_ui, pr_ui)
        All the predictions that have been made by a given algorithm
    n : int
        max number of predictions per user
    minimum_rating : float
        minimum rating threshold for an item

    Returns
    -------
        A dict where keys are user ids and values are lists of tuples:
    [(raw item id, rating estimation), ...] of size n.
    """

    # uid => [(166, 4.1196123), (356, 4.298811333)],
    # uid2 => [(166, 4.1196123), (356, 4.298811333)]
    top_n = defaultdict(list)

    for uid, iid, r_ui, pr_ui, _ in all_predictions:
        if (pr_ui >= minimum_rating):
            top_n[uid].append((iid, pr_ui))

    for uid, pr_ui_list in top_n.items():
        pr_ui_list.sort(key = lambda x: x[1], reverse = True)
        top_n[uid] = pr_ui_list[:n]

    return top_n