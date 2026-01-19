import numpy as np

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_variance(data):
    mean = calculate_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

def calculate_std(data):
    return calculate_variance(data) ** 0.5


def dataset_mean(matrix):
    return np.array([calculate_mean(matrix[:, i]) for i in range(matrix.shape[1])])

def dataset_std(matrix):
    return np.array([calculate_std(matrix[:, i]) for i in range(matrix.shape[1])])


def interclass_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)
