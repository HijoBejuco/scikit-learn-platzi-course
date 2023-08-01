import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')

    dt_features = dt_heart.drop(['target'], axis = 1)
    dt_target = dt_heart['target']

    #Splitting our data
    X_train, X_test, y_train, y_test = train_test_split(dt_features,
                                                        dt_target,
                                                        test_size=0.3,
                                                        random_state=42)
    
    #Scaling our data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    #Convencional PCA
    pca = PCA(n_components=3)
    pca.fit(X_train_scaled)

    #Incremental PCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train_scaled)

    #Let's plot a graph which tells us how much of the variance explain each component
    plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.show()

    #Apply pca to train and test datasets
    dt_train = pca.transform(X_train_scaled)
    dt_test = pca.transform(X_test_scaled)

    #lets model, logistic score is the function to evaluate the accuracy of predictions
    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train, y_train)
    print('score pca:', logistic.score(dt_test, y_test))

    
