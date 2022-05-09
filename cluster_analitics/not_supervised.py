from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemblr import IsolationForest
from minisom import MiniSom
from sklearn.cluster import DBSCAN




##########################################################
#               Clases no supervisadas                   #
########################################################## 
class IsolationForest_Custom(IsolationForest):

    def predict(self, X):
        pred = super().predict(X)
        scale_func = np.vectorize(lambda x: 1 if x == -1 else 0)
        return scale_func(pred)



class LocalOutlierFactor_Custom(LocalOutlierFactor):
    pass


class Som_Custom(MiniSom):
    pass



class Dbscan_Custom(DBSCAN):
    pass