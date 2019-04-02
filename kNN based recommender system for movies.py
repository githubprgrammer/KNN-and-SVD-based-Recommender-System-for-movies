from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import KNNBasic
from surprise.model_selection import KFold
from collections import defaultdict
from surprise import Dataset
from surprise import NormalPredictor
from surprise import SVD
import numpy as np


#load Movielens 100k Dataset
data = Dataset.load_builtin('ml-100k')

#function for computing precision and recall of predictions at cutoff k
def precision_recall_at_k(predictions, k=10, threshold=4):

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items: item is relevant if its true rating is greater than the treshold
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k: item is recommended, if its estimated rating is greater than treshold and if it is amongst the k highest estimated ratings
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant = recommendet items that are relevant/recommended items
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended = recommendet items that are relevant/relevant items
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

#function for computing the root mean square error
def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_

#function for computing the mean absolute error
def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_

#get the top-N recommendations for each user
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

#cross validation iterator
kf = KFold(n_splits=5)
algo_SVD = SVD()
algo_KNN = KNNBasic()

averaged_precision_combined = list()
averaged_recall_combined = list()
averaged_precision_SVD = list()
averaged_recall_SVD = list()
averaged_precision_KNN = list()
averaged_recall_KNN = list()
for trainset, testset in kf.split(data):
    #train the algorithm
    algo_SVD.fit(trainset)
    #get rating predictions for items from users
    predictions_SVD = algo_SVD.test(testset)
    algo_KNN.fit(trainset)
    predictions_KNN = algo_KNN.test(testset)
    #here we will store the combined predictions of the SVD and KNN ratings
    predictions_combined = list()

    for i in range(len(predictions_SVD)):
        #convert predictions into list because it is a tuple and therefore immutable
        L1 = list(predictions_SVD[i])
        #update estimated rating value by taking the average of the estimates of the prediction of SVD and KNN algorithm
        L1[3] = (L1[3] + predictions_KNN[i].est) / 2
        #after changes convert it back to a tuple
        T1 = tuple(L1)
        predictions_combined.append(T1)
    precisions_combined, recalls_combined = precision_recall_at_k(predictions_combined, k=5, threshold=4)
    precisions_SVD, recalls_SVD = precision_recall_at_k(predictions_SVD, k=5, threshold=4)
    precisions_KNN, recalls_KNN = precision_recall_at_k(predictions_KNN, k=5, threshold=4)

    # Precision and recall can then be averaged over all users
    averaged_precision_SVD.append(sum(prec for prec in precisions_SVD.values()) / len(precisions_SVD))
    averaged_recall_SVD.append(sum(rec for rec in recalls_SVD.values()) / len(recalls_SVD))
    averaged_precision_KNN.append(sum(prec for prec in precisions_KNN.values()) / len(precisions_KNN))
    averaged_recall_KNN.append(sum(rec for rec in recalls_KNN.values()) / len(recalls_KNN))
    averaged_precision_combined.append(sum(prec for prec in precisions_combined.values()) / len(precisions_combined))
    averaged_recall_combined.append(sum(rec for rec in recalls_combined.values()) / len(recalls_combined))


print("averaged_precision for SVD algorithm:")
print(sum(averaged_precision_SVD) / len(averaged_precision_SVD))
print("averaged_recall for SVD algorithm:")
print(sum(averaged_recall_SVD) / len(averaged_recall_SVD))
print("averaged_precision for KNN algorithm:")
print(sum(averaged_precision_KNN) / len(averaged_precision_KNN))
print("averaged_recall for KNN algorithm:")
print(sum(averaged_recall_KNN) / len(averaged_recall_KNN))
print("averaged_precision for combined algorithm:")
print(sum(averaged_precision_combined) / len(averaged_precision_combined))
print("averaged_recall for combined algorithm:")
print(sum(averaged_recall_combined) / len(averaged_recall_combined))

#We compare our results with a random Predictor
algo_random = NormalPredictor()
algo_random.fit(trainset)
prediction_random = algo_random.test(testset)


#compare rmse and mae of the different algorithms
rmse(predictions_SVD)
rmse(predictions_KNN)
rmse(predictions_combined)
rmse(prediction_random)

mae(predictions_SVD)
mae(predictions_KNN)
mae(predictions_combined)
mae(prediction_random)





