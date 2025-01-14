'''
PROBLEM 4: MNIST, 20 NG : Train and test KNN classification (supervised)
Your goal in this problem is to write your own K-nearest neighbor (KNN) classifier.

For each of the two datasets, now in matrix format and with pairwise similarity computed, train and test a KNN classifier. You are required to implement KNN classification model yourself, though you may use support libraries / data-structures for the neighbor searching.

You should partition the datasets into (say) an 80/10/10 training/testing/validation sets. Note that the actual "training" here consists of simply identifying nearest neighbors---unlike other common classifiers, there is no iterative or gradient-based procedure.

Report both training performance and testing performance. If using Python, you are encouraged (but not required) to write a scikit-learn compatible *estimator* class supporting a common API interface, e.g. *.fit(), *.predict(), *.transform(), etc. See https://scikit-learn.org/stable/developers/develop.html for more details.
'''