
import numpy as np
def centerCal(x):
    
    n = x.shape[0]
    val = np.zeros(x.shape[1])
    for i in range(n):
        val = np.add(val, x[i,:])
    return val/n


def distPoint(x, point):
    n = x.shape[0]
    dist = np.zeros(n)
    for i in n:
        dist[i] = np.linalg.norm(x[i,:]-point)

    return dist




