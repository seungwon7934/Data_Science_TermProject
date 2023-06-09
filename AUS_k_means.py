import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings(action='ignore')

optimal_Ks = []


# Preprocessing data: delete unused features
def preprocessing_data(data):
    data.drop(['Date','Location','MinTemp','WindGustDir','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Temp9am','Year','Month'], axis = 1, inplace = True)
    return data


# Visualize the correlation
def correlation_heatmap(data):

    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    # Plot the heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.show()


# Visualize how RainTomorrow is distributed across clusters
def rainTomorrow_matrix(data, rainTomorrow, labels):

    data['rainTomorrow'] = rainTomorrow
    data['target'] = rainTomorrow
    data['cluster'] = labels

    data['target'] = data['target'].map({0: 'No', 1: 'Yes'})
    data['cluster'] = data['cluster']
    data_clustering_result = data.groupby(['target', 'cluster'])['rainTomorrow'].count()
    print(data_clustering_result)

    data.drop(['target', 'cluster', 'rainTomorrow'], axis = 1, inplace = True)


def k_means_clustering(data, rainTomorrow):

    # Determining the optimal number of clusters using the Elbow method
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 10))
    visualizer.fit(data)
    # visualizer.show()
    
    # Perform K-means cluster analysis by selecting the optimal number of clusters
    optimal_k = visualizer.elbow_value_
    optimal_Ks.append(optimal_k)
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, max_iter=300)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Visualizing in numbers how the actual RainTomorrow values ​​are distributed
    rainTomorrow_matrix(data, rainTomorrow, labels)

    # Visualize clustering through PCA
    pca = PCA(n_components=2)
    pca_transformed_data = pca.fit_transform(data)
    pca_transformed_data = pd.DataFrame(pca_transformed_data, columns=["PCA_X", "PCA_Y"])
    pca_transformed_data = pd.DataFrame(pca_transformed_data, columns=["PCA_X", "PCA_Y"])
    pca_transformed_data['label'] = labels

    # Visualize the results of clustering and how RainTomorrow is distributed among those results
    plt.subplots(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data = pca_transformed_data, x = 'PCA_X',y = 'PCA_Y', hue = 'label')
    plt.title("Clustering Result")

    plt.subplot(1, 2, 2)
    pca_transformed_data['RainTomorrow'] = rainTomorrow 
    sns.scatterplot(data = pca_transformed_data, x = 'PCA_X',y = 'PCA_Y', hue = 'RainTomorrow');
    plt.title("Actual")
    plt.show()


def clustering_all_cases(data):
    rainTomorrow = data.iloc[:, -1]
    data.drop(['RainTomorrow'], axis = 1, inplace = True)
    for i in range(2, data.shape[1]):
        k_features = extract_random_attributes(data, i)
        for dataframe in k_features:
            print(dataframe.columns)
            k_means_clustering(dataframe, rainTomorrow)


def extract_random_attributes(dataframe, k):
    columns = dataframe.columns.tolist()  # List all properties of dataframe
    all_combinations = list(itertools.combinations(columns, k))  # Generate all possible attribute combinations

    all_dataframes = []
    for combination in all_combinations:
        new_dataframe = dataframe[list(combination)]  # Create a new dataframe with attributes corresponding to the combination
        all_dataframes.append(new_dataframe)

    return all_dataframes


def clustering_positive_case(data):
    rainTomorrow = data.iloc[:, -1]
    pov_data = data.drop(['MaxTemp','Pressure9am','Pressure3pm','Temp3pm'], axis = 1)
    print(pov_data)
    clustering_all_cases(pov_data)
    data.drop(['RainTomorrow'], axis = 1, inplace = True)
    k_means_clustering(pov_data, rainTomorrow)


def KMedoids_clustering(data):

    visualizer = KElbowVisualizer(model, k=(2, 10))
    visualizer.fit(data)
    optimal_k = visualizer.elbow_value_
    model = KMedoids(n_clusters=optimal_k, max_iter=300)

    rainTomorrow = data['RainTomorrow']
    data.drop(['RainTomorrow'], axis = 1, inplace = True)
    data.info()
    model.fit(data)
    labels = model.labels_

    # Visualize clustering through PCA
    pca = PCA(n_components=2)
    pca_transformed_data = pca.fit_transform(data)
    pca_transformed_data = pd.DataFrame(pca_transformed_data, columns=["PCA_X", "PCA_Y"])
    pca_transformed_data = pd.DataFrame(pca_transformed_data, columns=["PCA_X", "PCA_Y"])
    pca_transformed_data['label'] = labels

    # Visualize the results of clustering and how RainTomorrow is distributed among those results
    plt.subplots(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data = pca_transformed_data, x = 'PCA_X',y = 'PCA_Y', hue = 'label')
    plt.title("Clustering Result")

    plt.subplot(1, 2, 2)
    pca_transformed_data['RainTomorrow'] = rainTomorrow
    sns.scatterplot(data = pca_transformed_data, x = 'PCA_X',y = 'PCA_Y', hue = 'RainTomorrow')
    plt.title("Actual")
    plt.show()

    print_purity(labels= labels, rainTomorrow= rainTomorrow)


def DBSCAN_clustering(data):
    rainTomorrow = data['rainTomorrow']
    dbscan_data = data.drop(['rainTomorrow'], axis = 1)
    model = DBSCAN(min_samples=6)
    model.fit(dbscan_data)
    labels = model.labels_

    # Visualize clustering through PCA
    pca = PCA(n_components=2)
    pca_transformed_data = pca.fit_transform(data)
    pca_transformed_data = pd.DataFrame(pca_transformed_data, columns=["PCA_X", "PCA_Y"])
    pca_transformed_data = pd.DataFrame(pca_transformed_data, columns=["PCA_X", "PCA_Y"])
    pca_transformed_data['label'] = labels

    # Visualize the results of clustering and how RainTomorrow is distributed among those results
    plt.subplots(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data = pca_transformed_data, x = 'PCA_X',y = 'PCA_Y', hue = 'label')
    plt.title("Clustering Result")

    plt.subplot(1, 2, 2)
    pca_transformed_data['RainTomorrow'] = rainTomorrow
    sns.scatterplot(data = pca_transformed_data, x = 'PCA_X',y = 'PCA_Y', hue = 'RainTomorrow')
    plt.title("Actual")
    plt.show()

    print_purity(labels= labels, rainTomorrow= rainTomorrow)


def print_purity(labels, rainTomorrow):
    contingency_matrix = metrics.cluster.contingency_matrix(rainTomorrow, labels)
    purity =  np.sum(np.amax(contingency_matrix, axis = 0))/np.sum(contingency_matrix)
    
    print(purity)


def main():
    data = pd.read_csv('res\preprocessed_data.csv')     # Read data and store
    correlation_heatmap(data)   # Visualize correlation by Heatmap
    data = preprocessing_data(data)     # Delete not using feature 
    KMedoids_clustering(data)
    DBSCAN_clustering(data)
    correlation_heatmap(data)     
    clustering_all_cases(data)    # Using randomly Select features, make clusters and analyze and visualize clusters
    clustering_positive_case(data)  # When features are all positive correlation coefficient, over than 0.2


if __name__ == "__main__":
    data = main()