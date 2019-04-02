# KNN-and-SVD-based-Recommender-System-for-movies
In this project we will analyze the Movielens 100k Dataset which consists of 100.000 ratings from 1000 users on 1700 movies. All users in this dataset have at least rated 20 movies. Apart from this information, simple demographic information for the users like age, gender, occupation is included. The dataset can be obtained on the following permalink: http://grouplens.org/datasets/movielens/100k/
We will use a hybrid collaborative filtering approach where we will combine the results of the K Nearest Neighbor algorithm and the memory-based SVD algorithm to predict the movie ratings of the users. The advantage of the collaborative filtering algorithms is that no knowledge about item features is needed. So we can ignore the movie tags and the
demographic information and concentrate on the users and their ratings. We will evaluate the hybrid model to see if a combination between a modelbased (SVD) and a memory-based (KNN) approach delivers better results than the approaches on their own.

## Implementation
For the implementation of this project we have used “surprise” a Python scikit for recommender systems. It has predefined all major recommendation algorithms such as KNN, SVD, SVD++. We created a new hybrid algorithm by combining the results of KNN and SVD. On http://surpriselib.com/ you have access to the surprise library.

Hence, we first run SVD on the training data and get a model. Then we do the same with KNN. With KNN we implemented a user-based collaborative filtering model. To compute the similarity between the K nearest neighbor in the KNN algorithm we used cosine similarity. For both SVD and KNN we get predictions for the movie ratings of each user. The results are combined by averaging the estimated rating of KNN and SVD.

We used 5 cross-fold validation for splitting our data in train and testing sets. As evaluation metrics we used Root Mean Square Error, Mean Absolute Error and precision and recall. The precision and recall results of the 5 crossfold validation was averaged
for each algorithm (SVD, KNN, combination of SVD and KNN, random prediction).

## Results
Below you can see the results of the implementation:

```
averaged_precision for SVD algorithm:
0.8745
averaged_recall for SVD algorithm:
0.2583
averaged_precision for KNN algorithm:
0.8436
averaged_recall for KNN algorithm:
0.2739
averaged_precision for combined algorithm:
0.8734
averaged_recall for combined algorithm:
0.2565
RMSE of SVD: 0.9377
RMSE of KNN: 0.9763
RMSE of combined: 0.9354
RMSE of random: 1.5288
MAE of SVD: 0.7402
MAE of KNN: 0.7710
MAE of combined: 0.7421
MAE of random: 1.2294
```

## Evaluation
As we can observe, the SVD model outperforms KNN and the random predictor in all metrics. It has the smallest RMSE, MAE and recall and the highest precision. The KNN model is nearly as good as SVD. SVD is just 3.95 % better in RMSE, 3.99% better in MAE. Furthermore, SVD has a 3.94% higher precision and a 5.69 % better recall rate. Of course, both, KNN and SVD, are much better than the random prediction model. KNN for example has a 37.28% smaller MAE and a 36.14 % smaller RMSE than the random predictor, which is enormous. It should also be noted, that the difference for MAE and the difference for RMSE between the models is almost the same. For example, SVD is around 4% better in RMSE (the exact value is 3.95% as stated before) as well as in MAE (3.99%) than KNN. And KNN is around 36% better in RMSE and MAE than the random predictor. This closeness between RMSE and MAE may indicate, that these metrics are very similar and that one does not get any additional information by applying both metrics.

Now let us compare our combined model with the other models. Since SVD is the best of the single models, it is sufficient to just compare SVD with the combined model. In regard to RMSE, the combined approach is only 0.245% better than the SVD model. For MAE however, it’s the opposite, here the SVD model is 0.256% better than the combined approach. Regarding precision SVD has a 0.126% higher precision that the combined model. Also the recall rate of the SVD algorithm is 0.7% higher than that of the combined algorithm.

## Conclusion
Our combined model has a very high precision. This means that most of the recommended items are relevant. However, the model has also a relatively low recall, which means that the proportion of relevant items that are recommended is very small. The same applies for SVD and KNN. The results have shown, that the combined model, where we averaged the estimated ratings of the KNN and SVD model, is not significantly better than for example the SVD model alone. In fact, we can observe from the results,
that the SVD model performs much better than the KNN model on the 100k movielens dataset, such that, if we combine the models, the result for the combined model is in most metrics (MAE, precision and recall) slightly worse than for the SVD algorithm. Hence, the combination of the SVD and KNN model is not worth the effort and we would do better if we just used the SVD algorithm. As a model-based approach it is much faster than the KNN approach, because we have only to generate the model the first time and then can use this for new data points. For KNN however, for every new data point we have to run the whole algorithm again.
