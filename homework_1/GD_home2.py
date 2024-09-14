import sys

import numpy as np
import matplotlib.pyplot as plt

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.00001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.X = np.array([])
        self.Y = np.array([])
        self.m = 0
        self.b = 0


    def PlotData(self):
        plt.figure(figsize=(10, 6), dpi=150)
        plt.style.use('fivethirtyeight')
        plt.scatter(self.X, self.Y, color='black', s=30, label='Data')
        plt.plot(self.X, self.predict(), label="Predicted", color='blue', linewidth=1.5)
        plt.title("Gradient Descent Fit", fontsize=18)
        plt.xlabel("X", fontsize=14)
        plt.ylabel("Y", fontsize=14, rotation=0)
        plt.axhline(0, color='black',linewidth=1)
        plt.axvline(0, color='black',linewidth=1)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig('gradient_descent_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    def SaveData(self):
        np.savez('SecondDataSet.npz', array1=self.X, array2=self.Y)

    def GenerateData(self):
        loaded_data = np.load('SecondDataSet.npz')
        self.X = loaded_data['array1']
        self.Y = loaded_data['array2']

    def fit(self):
        m = 10
        b = 0
        n = self.X.shape[0]

        for _ in range(self.iterations):
            Y_pred = m * self.X ** 2 - b * self.X ** 3

            m_gradient = (-2 / n) * np.sum(self.X ** 2 * (self.Y - Y_pred))
            b_gradient = (2 / n) * np.sum(self.X ** 3 * (self.Y - Y_pred))

            m -= self.learning_rate * m_gradient
            b -= self.learning_rate * b_gradient

        self.m, self.b = m, b
        print(f"Optimized parameters: m = {self.m}, b = {self.b}")

    def predict(self):
        return self.m * self.X ** 2 - self.b * self.X ** 3

sys.stdout = open('result.txt', 'w')

clf = GradientDescentLinearRegression()
clf.GenerateData()
clf.fit()
clf.PlotData()

sys.stdout.close()