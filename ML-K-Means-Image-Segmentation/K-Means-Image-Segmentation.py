from cgitb import grey
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from datetime import datetime
from mpl_toolkits import mplot3d

# Ryan Filgas
# Computer Vision

run = 1


def get_distance(a, b, c, d, e, f):
    return pd.DataFrame((a-b)**2 + (c-d)**2 + (e-f)**2).pow(.5)


def one_point_distance(a, b, c, d, e, f):
    return np.sqrt((a-b)**2 + (c-d)**2 + (e-f)**2)


# Classify a point to a cluster. Return the index.
def classify(point, clusters, K):
    if K == 1:
        return 0
    else:
        return (get_distance(clusters[0], point[0], clusters[1], point[1], clusters[2], point[2])).idxmin()[0]


def one_point_distance(a, b, c, d, e, f):
    return np.sqrt((a-b)**2 + (c-d)**2 + (e-f)**2)


# within cluster sum of squares
def WCSS(cluster_buckets, solution, K):
    sum = 0
    for i in range(K):
        bucket = cluster_buckets[i]
        bucket_size = len(bucket.T)
        for j in range(bucket_size):
            sum += (one_point_distance(bucket[j][0],
                    solution[i][0], bucket[j][1], solution[i][1], bucket[j][2], solution[i][2])**2)
    return sum


# Kmeans for 1 K value
def kmeans_instance(R, K, data, cluster):
    data = pd.DataFrame(data)
    solution = cluster.copy()

    for i in range(R):
        #print("entered kmeans instance")
        # classify all points and add to temporary bucket
        temp = pd.DataFrame()
        cluster_buckets = [temp for i in range(K)]
        len_data = len(data)
        for j in range(len_data):
            c_index = classify(
                [data[0][j], data[1][j], data[2][j]], solution, K)
            cluster_buckets[c_index] = pd.concat(
                [cluster_buckets[c_index], data.T[j]], axis=1, ignore_index=True)
            # if j % 10000 == 0:
            #    print("classifications completed: ", j,
            #          " of ", len_data, " : ", j/len_data, "%")

        # update mean to output solution
        solution = solution.T
        for k in range(K):
            solution[k] = cluster_buckets[k].T.mean()
        solution = solution.T

        # Print iterations
        # error_calc = WCSS(cluster_buckets, np.array(solution), K)
        # plot_kmeans(K, R, np.array(solution), error_calc, cluster_buckets)

    # calculate error_calc
    error_calc = WCSS(cluster_buckets, np.array(solution), K)
    return np.array(solution), error_calc, cluster_buckets


def kmeans(R, K, data):
    clusters = list()
    for i in range(R):
        temp = np.array((data.sample(K)))
        clusters.append(temp)

    # run k-means here x 10 and store sum of squares error_calc
    errors = list()
    solutions = list()
    all_cluster_buckets = list()

    for i in range(R):
        solution, error_calc, cluster_buckets = kmeans_instance(
            R, K, data, pd.DataFrame(clusters[i]))
        errors.append(error_calc)
        solutions.append(solution)
        all_cluster_buckets.append(cluster_buckets)
        print("Instance complete.")

    solution_location = errors.index(min(errors))
    best_error = min(errors)
    solution_location = errors.index(best_error)
    best_solution = solutions[solution_location]
    best_clusters = all_cluster_buckets[solution_location]
    return best_solution, best_error, best_clusters


def plot_kmeans(K, R, best_solution, best_error, cluster_buckets):
    global run
    filename = str(K) + "-" + str(datetime.now()) + ".jpeg"
    points = list()
    classes = list()
    for i in range(K):
        bucket = cluster_buckets[i]
        bucket_size = len(bucket.T)
        for j in range(bucket_size):
            points.append(bucket[j])
            classes.append(i)
    points_x = np.array(points).T[0]
    points_y = np.array(points).T[1]
    points_z = np.array(points).T[2]
    title = "K = " + str(K) + ", Error = " + str(best_error)

    ax = plt.axes(projection='3d')
    ax.scatter3D(points_x, points_y, points_z, c=classes,
                 s=10, alpha=1, cmap='rainbow')
    if run % 2 == 0:
        ax.view_init(-140, 60)

    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    run += 1

    # plt.rcParams["figure.figsize"] = (10, 10, 10)
    # plt.scatter(points_x, points_y, points_z, c=classes,
    #             s=10, cmap='tab10')
    # plt.scatter(best_solution.T[0],
    #             best_solution.T[1], best_solution.T[0], c='black', s=100, alpha=1)
    # plt.title(title)
    # plt.savefig(filename)
    # plt.clf()


KVALS = [5, 10]
R = 10

filter1_og = cv2.imread("inputs/Kmean_img1.jpg", cv2.COLOR_BGR2RGB)
filter2_og = cv2.imread("inputs/Kmean_img2.jpg", cv2.COLOR_BGR2RGB)


filter1_flattened = pd.DataFrame(
    np.float64(filter1_og.reshape((-1, 3))))
filter2_flattened = pd.DataFrame(
    np.float64(filter2_og.reshape((-1, 3))))


for K in KVALS:
    best_solution, best_error, cluster_buckets = kmeans(
        R, K, filter1_flattened)
    plot_kmeans(K, R, best_solution, best_error, cluster_buckets)
    #print("Error for K =  ", K, ": ", best_error)

    best_solution, best_error, cluster_buckets = kmeans(
        R, K, filter2_flattened)
    plot_kmeans(K, R, best_solution, best_error, cluster_buckets)
    #print("Error for K =  ", K, ": ", best_error)

    #print("K complete: ", K)
