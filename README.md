 cs224w
 =
cs224w group project


CS224 Project Plan Outline
-

1. Run the Rev2 code to get a fairness score per user

    This will give a relative fairness score per user (lowest score means most likely to be bad actor, highest fairness score most likely to be good user)

1. Run ReFex (HW 2 question 2) to try and improve the fairness score from Rev2 algorithm

    1. Try different attributes in the ReFex node embedding vector (total 5 features)
        1. Traditional RoFex Embedding
           1.  Edge number, incoming #
            1. Edge number outcoming #
            1. ~~Edge different: incoming - outcoming~~
            1. egoNet edges  : total edges between members of the egonet
            1. edges between egoNet
        1. Rev2 features
            1. Rev2 fairness Score
            1. Rev2 fairness median score
        1. If we have time: Min, median, max, std of delta t for ratings transactions or ratings values (if have time, to include total transaction #, is this equaling to incoming + outcoming edges?)
        1. In the first iteration, we concatenate the mean,sum, std of all u's neighbors' feature vectors to ~ Vu, and do the same for sum

1. Use the vector embeddings from step 2 to create models that will label user good vs bad actor. Do this improve the accuracy of the Rev2 algorithm? (compare with ground truth, to measure the accuracy of the results from combination of Rev2+RoFex)
    1. Cosine similarity, L2 norm (find similar bad user, and compare with Rev and ground truth)
        1. Using ground truth, do the bad actors have vector embeddings that are more similar
    1. K-means (community)
    1. Do good vs bad users cluster?
    1. Neural Net, decision tree (classification)
        1. Can we create a model that can accurately label users using the vector embeddings as the input features

1. Node2Vec

    1. Do we notice any structural or community patterns between the bad actor nodes?

1. Puppet Master
    1. Bad users with same role, bad users’ direct neighbors,, or within same community should form a “puppet army”. Write some analysis about these users about their similarities, like rating burst, rating behavior, abnormal network structure.


Files
-----

**RoIX_embedding.py**

1. We first initialize each node with 6 features, 4 are network structure related, other two are from the Rev2 fairness.
1. Then we do 3 iteration, expand to totally 54 features for each node,
1. Finally dump the features in a csv file, ./results/%s_graph_embedding_vectors.csv.

As side product, also dump the who graph (with embedding features) in a pickle file
"./results/%s_graph_embedding_featured_graph.pkl"

We compare and remove some related features.

**cosin_similarity.py**

1. Load the embedding csv file, for one (manually pick one bad user id) user, calcualte the cosine similarity score with each of other nodes, sort the score from high to low.
1. Then write the results in a csv file, "./results/%s_%d_similarity_vectors.csv"
1. Pick top K and last K nodes, compare with the ground truth, calculate the intersection




**k-means.py**

One class to cluster nodes by k-means


**ground_truth_analysis.py**

1. Analysis the groun truth network file, "./rev2/data/%s_gt.csv"
1. Parse the file and save good user list and bad user list to two pickle file
    1. "./results/%s_gt_good_users_set.pkl"
    1. "./results/%s_gt_bad_users_set.pkl"

**networkx_utils.py**

Some function to draw diagram by networkx or pyplot

**draw_diagrams.py**

Classes where all the diagrams are drawn there.

**environment.yml**

Used by conda, setup the python dev environment

DataSet
-------

1. Rev2 paper alpha dataset contains
https://cs.stanford.edu/~srijan/rev2/

Note: alpha_network.pkl and alpha_network.csv have different nodes and edges. (not sure why)

Rev2 load the pkl file to load the graph, and in the final fairness dump file, it only contains the user nodes and some user nodes are missing. (3786 vs. 3286)

Our alternative solution is to read the original alpha_network.csv file,filter out all the nodes which has no fairness score.

And the ground truth file only contains around 200 labels, which only can be used for cross and spot checking. Not enough data for NN training.


