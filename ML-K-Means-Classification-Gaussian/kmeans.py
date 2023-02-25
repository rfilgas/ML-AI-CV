import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import e
import seaborn as sn
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix_from_data

# Ryan Filgas
# Machine learning



def get_distance(a,b,c,d):
    return pd.DataFrame((a-b)**2 + (c-d)**2).pow(.5)



def one_point_distance(a,b,c,d):
    return np.sqrt(((a-b)**2 + (c-d)**2))



# Classify a point to a cluster. Return the index.
def classify(point, clusters, K):
    if K ==1:
        return 0
    else:
        return (get_distance(clusters[0], point[0], clusters[1], point[1])).idxmin()[0]


#within cluster sum of squares
def WCSS(cluster_buckets, solution, K):
    sum = 0
    for i in range(K):
        bucket = cluster_buckets[i]
        bucket_size = len(bucket.T)
        for j in range(bucket_size):
            sum += (one_point_distance(bucket[j][0], solution[i][0], bucket[j][1], solution[i][1])**2)
    return sum


# Kmeans for 1 K value
def kmeans_instance(R,K,data,cluster):
    data = pd.DataFrame(data)
    solution = np.array(cluster).copy()
    
    for i in range(R):
        #classify all points and add to temporary bucket
        temp = pd.DataFrame()
        cluster_buckets = [temp for i in range(K)]

        for j in range(len(data)):
            c_index = classify([data[0][j], data[1][j]], cluster, K)
            cluster_buckets[c_index] = pd.concat([cluster_buckets[c_index], data.T[j]], axis=1, ignore_index=True)
        
        #update mean to output solution
        for k in range(K):
            solution[k] = cluster_buckets[k].T.mean()

    #calculate error_calc
    error_calc = WCSS(cluster_buckets, solution, K)
    return solution, error_calc, cluster_buckets



def kmeans(R, K, data):
    clusters = list()
    for i in range(R):
        temp = np.array((data.sample(K)))
        clusters.append(temp)

    #run k-means here x 10 and store sum of squares error_calc
    errors = list()
    solutions = list()
    all_cluster_buckets = list()

    for i in range(R):
        solution, error_calc, cluster_buckets = kmeans_instance(R,K,data,pd.DataFrame(clusters[i]))
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
    filename = str(K) + ".jpeg"
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
    title = "K = " + str(K) + ", Error = " + str(best_error)

    plt.rcParams["figure.figsize"] = (10,10);
    plt.scatter(points_x, points_y, c=classes,
            s=10, cmap='tab10');
    plt.scatter(best_solution.T[0], best_solution.T[1], c='black', s=100, alpha=1);
    plt.title(title);
    plt.savefig(filename);
    plt.clf()



KVALS = [1,2,3,4,5,6,7,8,9,10]
R = 10
data= pd.DataFrame(pd.read_csv('cluster_dataset.txt', sep="  ", header=None))

for K in KVALS:
    best_solution, best_error, cluster_buckets = kmeans(R,K,data)
    print("Error for K =  ", K, ": ", best_error)
    plot_kmeans(K, R, best_solution, best_error, cluster_buckets)
    print("K complete: ", K)
