import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns 
from plotly.subplots import make_subplots


def compPlot(x, y, y_pred,*,engine='matplotlib',dim = 2, color=['blue', 'red']):
    if dim == 2:    
        if engine == 'matplotlib':
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
            ax1.scatter(x[y==0,0],x[y==0,1], color = color[0])
            ax1.scatter(x[y==1,0],x[y==1,1], color = color[1])
            ax1.set_title('Valores originales')

            ax2.scatter(x[y_pred==0,0],x[y_pred==0,1], color = color[0])
            ax2.scatter(x[y_pred==1,0],x[y_pred==1,1], color = color[1])
            ax2.set_title('Valores obtenidos')
            plt.show()
        
        if engine == 'seaborn':
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
            sns.scatterplot(ax = ax1, x=x[:,0],y=x[:,1], hue=y)
            ax1.set_title('Valores originales')

            sns.scatterplot(ax = ax2, x=x[:,0],y=x[:,1], hue=y_pred)
            ax2.set_title('Valores obtenidos')
            plt.show()

        if engine == 'plotly':
            fig = make_subplots(rows=1, cols=2)
            t = y==0
            n = y==1
            fig.add_trace(go.Scatter(x=x[t,0],y=x[t,1], mode='markers', marker=dict(color=color[0]), 
                                       name='No fraude'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x[n,0],y=x[n,1], mode='markers', marker=dict(color=color[1]), 
                                       name='Fraude'), row=1, col=1)
            t = y_pred==0
            n = y_pred==1
            fig.add_trace(go.Scatter(x=x[t,0],y=x[t,1], mode='markers', marker=dict(color=color[0]),
                                       name = 'No fraude'), row=1, col=2)
            fig.add_trace(go.Scatter(x=x[n,0],y=x[n,1], mode='markers', marker=dict(color=color[1]), 
                                       name = 'Fraude'), row=1, col=2)
            fig.show()
    if dim == 3:
        if x.shape[1] < 3:
            print('Dimensiones de x insuficientes para plotear en 3D')
            raise ValueError
        if engine != 'plotly':
            print('Los plots en 3D se realizan en plotly')
        fig = make_subplots(rows=1, cols=2,
                   specs=[[{'type': 'scene'}, {'type': 'scene'}]])
        t = y==0
        n = y==1
        fig.add_trace(go.Scatter3d(x=x[t,0],y=x[t,1],z=x[t,2], mode='markers', marker=dict(color=color[0]), 
                                   name='No fraude'), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=x[n,0],y=x[n,1],z=x[n,2], mode='markers', marker=dict(color=color[1]), 
                                   name='Fraude'), row=1, col=1)
        t = y_pred==0
        n = y_pred==1
        fig.add_trace(go.Scatter3d(x=x[t,0],y=x[t,1],z=x[t,2], mode='markers', marker=dict(color=color[0]),
                                   name = 'No fraude'), row=1, col=2)
        fig.add_trace(go.Scatter3d(x=x[n,0],y=x[n,1],z=x[n,2], mode='markers', marker=dict(color=color[1]), 
                                   name = 'Fraude'), row=1, col=2)
        fig.show()

def precisionRecall(precision, recall,*,engine='matplotlib',color=['blue', 'red']):
    if engine == 'matplotlib':
        for i,pr in enumerate(zip(precision,recall)):
            pre,rec = pr
            fig = plt.figure(figsize=(10,8))
            plt.bar(i,pre, color=color[0])
            plt.bar(i,rec, color=color[1])
        plt.show()
        
    if engine == 'seaborn':
        for i,pr in enumerate(zip(precision,recall)):
            pre,rec = pr
            fig = plt.figure(figsize=(10,8))
            sns.barplot(i,pre, color=color[0])
            sns.barplot(i,rec, color=color[1])
        plt.show()
    if engine == 'plotly':
        fig = go.Figure(data = [
            go.Bar(name='Precision',x = np.arange(len(precision)), y=precision),
            go.Bar(name='Recall',x = np.arange(len(recall)), y=recall)
        ])
        fig.update_layout(barmode='group')
        fig.show()

