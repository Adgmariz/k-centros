import numpy as np

def calc_minkowski_distance(x,y,p):
    """Calculates the Minkowski distance between x and y.
    Args:
        x(ndarray): a vector of length d
        y(ndarray): a vector of length d
        p(int): the order of the Minkowski distance
    """
    return np.power(np.sum(np.power(np.abs(x-y),p)),1/p)

def calc_distance_matrix(X, p):
    """Calculates distance_matrix.
    Args:
        X(ndarray): (n,d) matrix
        p(int): the order of the Minkowski distance
    """
    n = X.shape[0]
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i,j] = calc_minkowski_distance(X[i],X[j],p)
    return distance_matrix

def farthest_point_from_centers(distance_matrix, centers):
    #Calculates the point that is farthest from the points in centers.
    n = distance_matrix.shape[0]
    max_distance = -1
    farthest_point = -1
    for i in range(n):
        if i not in centers:
            distance = 0
            for center in centers:
                distance += distance_matrix[i,center]
            if distance > max_distance:
                max_distance = distance
                farthest_point = i
    if farthest_point == -1:
        raise Exception("farthest_point == -1")
    return farthest_point

def k_centers(data, k, p):
    """Calculates k-centers.
    Args:
        data(ndarray): (n,d) matrix
        k(int): the number of centers
        p(int): the order of the Minkowski distance
    """
    n = data.shape[0]
    distance_matrix = calc_distance_matrix(data,p)
    if k >= n:
        return np.arange(n)
    centers = []
    centers.append(np.random.randint(n-1))
    while len(centers) < k:
        centers.append(farthest_point_from_centers(distance_matrix,centers))
    return np.array(centers)