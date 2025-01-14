'''
PROBLEM 3 MNIST, 20 NG Preprocessing
Your goal in this problem is to *parse*, *normalize*, and otherwise *prepare* two common data sets (MNIST + 20NG) for classification. In this problem, that includes prepping the datasets for **[dis]similarity** computations.
Your first task is parsing. As this is the first assignment and as the parsers are very different for the two datasets (images vs. text), you may use any library/package to aid in the parsing here, however you are encouraged to write your own.

Your second task is normalization. The type of normalization used depends on the task and dataset. Common types of normalization include:
Shift-and-scale normalization: subtract the minimum, then divide by new maximum. Now all values are between 0-1
Zero mean, unit variance : subtract the mean, divide by the appropriate value to get variance=1
Term-Frequency (TF) weighting : map each term in a document with its frequency (text only; see the wiki page) It is up to you to determine the appropriate normalization.

Your final task is to compute several types of pairwise similarities, for use in the next question. You are encouraged to write your own implementation of the pairwise similarity/distance matrix computation---but unless explicitly specified we will accept any code using a common library available in Matlab/Java/Python/R.
Distance/similarity options to implement:
euclidian distance (required, library)
euclidian distance (required, your own - use batches if you run into memory issues)
edit distance (required for text) -or- cosine similarity (required for vectors)
jaccard similarity (optional)
Manhattan distance (optional)

Some tips / hints:
For 20NG, try TF normalization on e.g. the rows. Note for text is critical to maintain a sparse format due to large number of columns
Make sure any value transformation retains the 0 values.
As MNIST is comprised of pixel images, they often come 'normalized' in a pre-formatted range [0-255], however their are advantages to having 0 mean.
Do not normalize labels.
When normalizing a column, make sure to normalize its values across all datapoints (train, test, validation, etc) for consistency Depending on what similarity/distance measure you use, computation of similarity might be easy but the size of the similarity matrix might present a challenge.

Useful links include:
https://en.wikipedia.org/wiki/Feature_scaling
https://en.wikipedia.org/wiki/Distance_matrix
https://en.wikipedia.org/wiki/Distance
https://en.wikipedia.org/wiki/Category:Similarity_and_distance_measures
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.154.8446&rep=rep1&type=pdf
http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
'''