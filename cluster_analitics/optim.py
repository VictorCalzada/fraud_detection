from .analitics import ReductionDim, Analisis 
from itertools import product
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
"""
Reduction:
    PCA       -->  {n_components: (0-30)}
    encoder   -->  {capas:[], finalDim: (0-30), batch_size: int, epochs: int}

Modelos
    LOF       -->  {n_neighbors: int, contamination: frac}
    isoForest -->  {contamination: frac}
    som       -->  {sigma: (0.1-0.99), learning_rate: 0.2, neighborhood_function: 'gaussian', outliers_percentage: frac }
"""



def crossParam(param: dict):
    valores = list(param.values())
    newParam = []
    if len(valores) == 1:
        return valores[0]
    for i in valores:
        if type(i) != list:
            i = [i]
        newParam.append(i)
    return list(product(*newParam))


def reductionAnalisis(x, reduccion, analisis):
    reduc = reduccion.keys()
    redList = []
    anali = analisis.keys()
    anList = []
    for red in reduc:
        params = crossParam(reduccion[red])
        for par in params:
            aux = ReductionDim(red,x)
            aux.initParams(par)
            redList.append(aux)
    for an in anali:
        params = crossParam(analisis[an])
        for par in params:
            aux =  Analisis(an)
            aux.initParams(par)
            anList.append(aux)
    return redList, anList


@dataclass
class ModelosItems:
    reduccion: ReductionDim
    analisis: Analisis
    val: float = 0.0
    metric: str = 'precision'

    @classmethod
    def yVal(cls, y):
        cls.y = y
    
    @classmethod
    def setMetric(cls, metric):
        cls.metric = metric

    

    def calMetricVal(self):
        self.analisis.setX(self.reduccion.predict())
        y_pred = self.analisis.fit_predict()
        self.cm = confusion_matrix(self.y, y_pred)
        self.calVal()
        
    def precision(self):
        p,n = self.cm
        tp,fp,fn,tn = *p,*n
        return tn/(tn+fp)

    def recall(self):
        p,n = self.cm
        tp,fp,fn,tn = *p,*n
        return tn/(tn+fn)

    def calVal(self):
        if self.metric == 'precision':
            self.val = self.precision()
        elif self.metric == 'recall':
            self.val = self.recall()
    
    def __lt__(self, other):
        return self.val < other.val 

    def __le__(self, other):
        return self.val <= other.val 
    
    def __eq__(self, other):
        return self.val == other.val 
    
    def __ne__(self, other):
        return self.val != other.val 

    def __ge__(self, other):
        return self.val >= other.val 
    
    def __gt__(self, other):
        return self.val > other.val 
    
    def __str__(self):
        info = str(self.reduccion)+'\n'+str(self.analisis)
        return info

def main(x,y,reduccion = None, analisis= None):
    if reduccion ==  None and analisis==None:
        reduccion={'pca':{'finalDim':[2,3,4]}, 
               'encoder':{'capas':[[512,128], [32,16,8]], 'finalDim':[2,3,4], 'batch_size':20, 'epochs':10 }}

        analisis = {'lof':{'n_neighbors':[20,40],'contamination':[0.1,0.01]},
               'isoForest':{'contamination':[0.1,0.01]},
               'som':{'sigma':[0.5,0.99], 'learning_rate':[0.2,0.5], 'neighborhood_function':'gaussian', 'outliers_percentage':0.15}}
    
    red, an = reductionAnalisis(x, reduccion, analisis)
    redAn = list(product(red,an))
    obj = []
    for i in redAn:
        obj.append(ModelosItems(*i))
    for i in red:
        i.fit()

    ModelosItems.yVal(y)
    for o in obj:
        o.calMetricVal()
    obj.sort(reverse=True)
    return obj








        
    