import numpy as np
import matplotlib.pyplot as plt
import random

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.00001, iterations=10):
        self.learning_rate, self.iterations = learning_rate, iterations
        self.X = np.array([])
        self.Y = np.array([])
        self.m = 0
        self.b = 0
        self.k = 0

    def PlotData(self):
        plt.style.use('fivethirtyeight')
        plt.scatter(self.X, self.Y, color='black')
        plt.plot(self.X, self.predict())
        plt.show()

    def SaveData(self):
        np.savez('SecondDataSet.npz', array1=self.X, array2=self.Y)


    def GenerateData(self):
            loaded_data = np.load('SecondDataSet.npz')
            # Access individual arrays by their names
            self.X = loaded_data['array1']
            self.Y = loaded_data['array2']


    def fit(self):
        m = 10
        b = 0
        n = self.X.shape[0]
        for _ in range(self.iterations):
            #дописать алгоритм
            pass
        self.m, self.b = m, b
        print(" answer : ", self.m, self.b)

    def predict(self):
            return self.m * self.X**2 + self.b*self.X**3





clf = GradientDescentLinearRegression()
clf.GenerateData()
clf.fit()
clf.PlotData()



