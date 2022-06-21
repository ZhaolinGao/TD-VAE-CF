from sklearn import linear_model
import numpy as np
from tqdm import tqdm


class KAVgenerator:
    def __init__(self, positive_embeddings, negative_embeddings):
        '''Takes in the embeddings for the postivie political spectrum dimension,
        as well as the negative political spectrum dimension embeddings'''
        self.positive_embeddings = positive_embeddings
        self.negative_embeddings = negative_embeddings
        self.num_positive = self.positive_embeddings.shape[0]
        self.num_negative = self.negative_embeddings.shape[0]
        self.dim_activate = self.positive_embeddings.shape[1]
    
    def _get_cav(self, num_neg=1,num_vec=100):
        vectors = []
        if self.num_positive == 0 or self.num_negative == 0:
            #activation vectors are all zero vector
            return np.zeros(shape=(num_vec, self.dim_activate))

        for _ in range(num_vec):
            # Get positive and negative sample id's, and their vector embeddings
            positive_samples = np.random.choice(list(range(self.positive_embeddings.shape[0])), num_neg)
            negative_samples = np.random.choice(list(range(self.negative_embeddings.shape[0])), num_neg)
            v_positive_samples = self.positive_embeddings[positive_samples]
            v_negative_samples = self.negative_embeddings[negative_samples]

            X = np.vstack((v_positive_samples,v_negative_samples))
            Y = [1]*len(positive_samples) + [0]*len(negative_samples)
            lm = linear_model.LogisticRegression()
            lm.fit(X, Y)
            vectors.append(lm.coef_[0])
        return self.normalize_rows(np.vstack(vectors))

    def get_all_cav(self, num_negatives,num_cav):
        ret = []
        print("Generate Activation Vector")
        # get the directional vector for this component
        kavs = self._get_cav(num_negatives, num_cav)
        ret.append(kavs)
        #kp by sample by dim
        return np.stack(ret,axis=0)
        # return kavs

    def get_all_mean_cav(self, num_negatives, num_cav):
        all_cav = self.get_all_cav(num_negatives, num_cav)
        # print(all_cav.shape)
        return np.mean(all_cav, axis=1)

    def normalize_rows(self, x):
        #return x
        """
        function that normalizes each row of the matrix x to have unit length.
        Args:
        ``x``: A numpy matrix of shape (n, m)
        Returns:
        ``x``: The normalized (by row) numpy matrix.
        """
        return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
