import numpy as np
from sklearn.mixture import GaussianMixture


def cluster_by_time(df):
    # Reshape the data since GMM expects a 2D array
    X = df['edited_general_time'].values.reshape(-1, 1)

    # Determine the optimal number of clusters using Bayesian Information Criterion (BIC)
    n_components = np.arange(1, 10)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
    bics = [m.bic(X) for m in models]

    # Select the model with the lowest BIC
    best_n = n_components[np.argmin(bics)]
    gmm = GaussianMixture(n_components=best_n, covariance_type='full', random_state=0)
    gmm.fit(X)
    clusters = gmm.predict(X)

    # Add cluster labels to the dataframe
    df['time_cluster'] = clusters

    return df

