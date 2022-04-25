import numpy as np
import pandas as pd

import matplotlib as plt

# Reduccion de la dimensionalidad
from keras.models import Sequential, Model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import random as rn

from sklearn.decomposition import PCA

# Modelos
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from minisom import MiniSom
from sklearn.cluster import DBSCAN


class ReductionDim():
    def __init__(self, reduction: str, x: np.array = None):
        self.reductionN = reduction
        self.x = x
        self._redImplement()  # Declaracion del modelo de redcuccion de dimensionalidad a utilizar

    def setX(self, x: np.array):
        """
        Introducitr o cambiar valores de x
        """
        self.x = x

    def setReduction(self, reduction: str):
        """
        Cambiar reduccion
        """
        self.reductionN = reduction
        self._redImplement()

    def _redImplement(self) -> None:
        """
        Seleccion de la reduccion que se va a ejecutar 
        """
        def pca(n_components):
            return PCA(n_components=n_components)

        def encoder(capas, inputShape: int,  finalDim: int = 2):
            rn.set_seed(999)
            m = Sequential()

            for i, capa in enumerate(list(capas)):
                if i == 0:
                    m.add(
                        Dense(
                            capa, activation='elu',
                            input_shape=(inputShape,)))
                else:
                    m.add(Dense(capa, activation='elu'))
            m.add(Dense(finalDim, activation='linear', name='bottleneck'))

            l = len(capas)

            for i in range(l):
                o = l - i - 1
                m.add(Dense(capas[o], activation='elu'))
            m.add(Dense(inputShape, activation='sigmoid'))

            return m

        if self.reductionN == 'pca':
            self.reduction = pca

        if self.reductionN == 'encoder':
            self.reduction = encoder

    def initParams(self, args):
        """
        Inilizacion de los parametros necesarios para la reduccion
        """
        if self.reductionN == 'pca':
            self.finalDim = args
            self.reduction = self.reduction(n_components=self.finalDim)

        if self.reductionN == 'encoder':
            self._redImplement()
            rn.set_seed(999)
            self.capas, self.finalDim, self.batch_size, self.epochs = args
            self.inputShape = self.x.shape[1]
            self.reduction = self.reduction(
                self.capas, self.inputShape, self.finalDim)

    def fit(self):
        """
        Fit de los valores de x   
        """
        if self.x is None:
            raise ValueError

        if self.reductionN == 'pca':
            self.reduction.fit(self.x)

        if self.reductionN == 'encoder':
            rn.set_seed(999)
            self.reduction.compile(loss='mean_squared_error', optimizer=Adam())
            self.reduction.fit(
                self.x, self.x, batch_size=self.batch_size, epochs=self.epochs,
                verbose=1, validation_data=(self.x, self.x))

            self.reductionEnco = Model(
                self.reduction.input, self.reduction.get_layer('bottleneck').output)

    def predict(self):
        """
        Prediccion de los valores de x reducidos 
        """
        if self.x is None:
            raise ValueError

        if self.reductionN == 'pca':
            return self.reduction.transform(self.x)

        if self.reductionN == 'encoder':
            rn.set_seed(999)
            # bottleneck representation
            return self.reductionEnco.predict(self.x)

    def fit_predict(self):
        """
        Fit y prediccion de los valores de x reducidos 
        """
        if self.x is None:
            raise ValueError

        if self.reductionN == 'pca':
            self.reduction.fit(self.x)
            return self.reduction.transform(self.x)

        if self.reductionN == 'encoder':
            rn.set_seed(999)
            self.reduction.compile(loss='mean_squared_error', optimizer=Adam())
            self.reduction.fit(
                self.x, self.x, batch_size=self.batch_size, epochs=self.epochs,
                validation_data=(self.x, self.x))

            self.reductionEnco = Model(
                self.reduction.input, self.reduction.get_layer('bottleneck').output)
            # bottleneck representation
            return self.reductionEnco.predict(self.x)

    def defImplement(self):
        """
        Implementacion por defecto para ahorrar tiempo 
        """
        if self.x is None:
            raise ValueError

        if self.reductionN == 'pca':
            n_com = 2
            self.initParams(args=n_com)
            return self.fit_predict()

        if self.reductionN == 'encoder':
            capas = [512, 128]
            finalDim = 2
            batch_size = 20
            epochs = 1
            args = [capas, finalDim, batch_size, epochs]
            self.initParams(args=args)
            return self.fit_predict()

    def __str__(self):
        info = f'{"_"*75}\nReduccion --> {self.reductionN}\n'
        if self.reductionN == 'pca':
            info += f'\tNumero de parametros final --> {self.finalDim}\n'
        if self.reductionN == 'encoder':

            info += f'\tNumero de parametros final --> {self.finalDim}\n\tTamaño del batch --> {self.batch_size}\n\tNumero de epocas --> {self.epochs}\n'
            self.reduction.summary()
        info += '_'*75
        return info


class Analisis():
    def __init__(self, modeloN: str, x: np.array = None):

        self.modeloN = modeloN  # Modelo que se va a ejecutar
        self.x = x
        self._modImplement()  # Declaracion del modelo a utilizar
        if self.x is None:
            self.inputShape = 1
        else:
            self.inputShape = self.x.shape[1]

    def setX(self, x: np.array):
        self.x = x
        self.inputShape = self.x.shape[1]
        if self.modeloN == 'som':
            args = self.sigma, self.learning_rate, self.neighborhood_function, self.outliers_percentage, self.epochs
            self.initParams(args)

    def _modImplement(self) -> None:
        def lof(n_neighbors, contamination, novelty):
            return LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination,
                novelty=novelty)

        def isoForest(contamination):
            return IsolationForest(random_state=1, contamination=contamination)

        def som(sigma, learning_rate, neighborhood_function, n1=2, n2=1):
            return MiniSom(
                n1, n2, input_len=self.inputShape, sigma=sigma,
                learning_rate=learning_rate,
                neighborhood_function=neighborhood_function)

        def dbscan(eps, min_samples):
            return DBSCAN(eps=eps, min_samples=min_samples)

        if self.modeloN == 'lof':
            self.modelo = lof

        if self.modeloN == 'isoForest':
            self.modelo = isoForest

        if self.modeloN == 'som':
            self.modelo = som

        if self.modeloN == 'dbscan':
            self.modelo = dbscan

    def initParams(self, args):

        if self.modeloN == 'lof':
            self._modImplement()
            self.n_neighbors, self.contamination = args
            if self.contamination == -1:
                self.contamination = 'auto'
            self.modelo = self.modelo(
                self.n_neighbors, self.contamination, novelty=True)

        if self.modeloN == 'isoForest':
            self._modImplement()
            self.contamination = args
            if self.contamination == -1:
                self.contamination = 'auto'
            self.modelo = self.modelo(self.contamination)

        if self.modeloN == 'som':
            self._modImplement()
            self.sigma, self.learning_rate, self.neighborhood_function, self.outliers_percentage, self.epochs = args
            self.modelo = self.modelo(
                self.sigma, self.learning_rate, self.neighborhood_function)

        if self.modeloN == 'dbscan':
            self._modImplement()
            self.eps, self.min_samples = args
            self.modelo = self.modelo(self.eps, self.min_samples)

    def fit(self):
        if self.modeloN == 'som':
            self.modelo.train(self.x, self.epochs)
        else:
            self.modelo.fit(self.x)

    def predict(self) -> np.array:
        if self.modeloN == 'lof':
            y = self.modelo.predict(self.x)
            y[y == 1] = 0
            y[y == -1] = 1
            return y
        if self.modeloN == 'isoForest':
            y = self.modelo.predict(self.x)
            y[y == 1] = 0
            y[y == -1] = 1
            return y

        if self.modeloN == 'som':
            quantization_errors = np.linalg.norm(
                self.modelo.quantization(self.x) - self.x, axis=1)
            error_treshold = np.percentile(quantization_errors,
                                           100*(1-self.outliers_percentage)+5)
            is_outlier = quantization_errors > error_treshold
            y = [int(i) for i in is_outlier]
            return y
        if self.modeloN == 'dbscan':
            y = self.modelo.predict(self.x)
            y[y != -1] = 0
            y[y == -1] = 1
            return y

    def fit_predict(self) -> np.array:
        if self.modeloN == 'lof':
            y = self.modelo.fit_predict(self.x)
            y[y == 1] = 0
            y[y == -1] = 1
            return y
        if self.modeloN == 'isoForest':
            y = self.modelo.fit_predict(self.x)
            y[y == 1] = 0
            y[y == -1] = 1
            return y

        if self.modeloN == 'som':
            self.modelo.train(self.x, self.epochs)
            quantization_errors = np.linalg.norm(
                self.modelo.quantization(self.x) - self.x, axis=1)
            error_treshold = np.percentile(quantization_errors,
                                           100*(1-self.outliers_percentage)+5)
            is_outlier = quantization_errors > error_treshold
            y = [int(i) for i in is_outlier]
            return y

        if self.modeloN == 'dbscan':
            y = self.modelo.fit_predict(self.x)
            y[y != -1] = 0
            y[y == -1] = 1
            return y

    def __str__(self):
        info = f'{"_"*75}\nAnalisis --> {self.modeloN}\n'
        if self.modeloN == 'lof':
            info += f'\tVecinos --> {self.n_neighbors}\n\tContaminacion --> {self.contamination}\n'
        if self.modeloN == 'isoForest':
            info += f'\tContaminacion --> {self.contamination}\n'
        if self.modeloN == 'som':
            info += f'\tSigma --> {self.sigma}\n\tLearning Rate --> {self.learning_rate}\n\tFuncion de vecinos --> {self.neighborhood_function}\n\tPorcentage de outliers --> {self.outliers_percentage}\n\tEpocas --> {self.epochs}\n'
        if self.modeloN == 'dbscan':
            info += f'\tEps --> {self.eps}\n\tMin_samples --> {self.min_samples}\n'
        info += '_'*75
        return info
