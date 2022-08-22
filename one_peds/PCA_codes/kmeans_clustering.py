from matplotlib import projections
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def normalize_data(data_, n_comps):

    pca_ = PCA(n_comps)  # No of dimensions 次元数

    data_frame = pd.DataFrame(data_)
    #dataScaled = preprocessing.normalize(data_)
    dataScaled = preprocessing.normalize(data_frame)

    df_ = pca_.fit_transform(dataScaled)
    print(df_)
    print(len(df_))
    print(data_)
    return df_


def Plot_Kmeans(df_, no_of_clus, mode, data):
    kmeans = KMeans(no_of_clus)

    label = kmeans.fit_predict(df_)
    # print(label)
    t = np.zeros((680, 1))

    for i in range(len(t)-1):
        t[i] = t[i-1]+0.01

    print('t', t)
    print(t.shape)
    print(t.size)

    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    # print(u_labels)
    print('label', label)
    print(len(label))
    np.save("u_labels.npy", u_labels)
    print(centroids)

    r_d = data['relative_distance_index'].values
    r_t = data['relative_angle_index'].values
    r_v = data['relative_velocity_index'].values
    action = data['action_index'].values
    Q = data['q_value'].values

    if mode == "2d":
        for i in u_labels:
            # plt.scatter(df_[label == i, 0], df_[label == i, 1], label=i)
            plt.scatter(r_d[label == i], Q[label == i], label=i)
        # plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.ylim(-1, 1)
        plt.legend()
        # plt.savefig(
        #     "Tensorforce/Q_Learning/environments/everyStepQ/Two_peds/Expt6_200/Graphs/2D_Kmeans.jpg")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

    elif mode == "3d":
        fig = plt.figure(figsize=(20, 16))
        ax = plt.axes(projection="3d")
        for i in u_labels:
            # ax.scatter3D(df_[label == i, 0], df_[label == i, 1],
            #              df_[label == i, 2], label == i)
            ax.scatter3D(r_t[label == i], action[label == i],
                         Q[label == i], label == i)
        # ax.scatter(centroids[:, 0], centroids[:, 1],
        #            centroids[:, 2], s=80, color='k')
        ax.set_title(
            'Plot of Q values vs relative distance index vs relative_angle', fontsize=16)
        ax.set_xlabel('relative_distance_index', fontsize=16)
        ax.set_ylabel('action_index', fontsize=16)
        ax.set_zlabel('Q_value', fontsize=16)
        ax.legend(['0', '1', '2'])
        ax.zaxis.set_tick_params(labelsize=16)

        #ax.ylim(-1, 1)
        # plt.savefig(
        #     "Tensorforce/Q_Learning/environments/everyStepQ/Two_peds/Expt6_200/Graphs/3D_Kmeans.jpg")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()


def main():
    data = pd.read_csv(
        'Tensorforce/Q_Learning/environments/everyStepQ/Expt37_300/Data/q_table.csv')

    num_components = 5
    num_clusters = 3
    varNames = data.columns

    mode = "3d"

    dataSet = normalize_data(data, num_components)
    Plot_Kmeans(dataSet, num_clusters, mode, data)


main()
