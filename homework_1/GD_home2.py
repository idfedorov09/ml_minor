import sys
import time

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

    def PlotData(self, filename: str = 'gradient_descent_plot'):
        plt.figure(figsize=(10, 6), dpi=150)
        plt.style.use('fivethirtyeight')
        plt.scatter(self.X, self.Y, color='black', s=30, label='Data')
        plt.plot(self.X, self.predict(), label="Predicted", color='blue', linewidth=1.5)
        plt.title("Gradient Descent Fit", fontsize=18)
        plt.xlabel("X", fontsize=14)
        plt.ylabel("Y", fontsize=14, rotation=0)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
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
        errors = []

        for _ in range(self.iterations):
            Y_pred = m * self.X ** 2 - b * self.X ** 3

            m_gradient = (-2 / n) * np.sum(self.X ** 2 * (self.Y - Y_pred))
            b_gradient = (2 / n) * np.sum(self.X ** 3 * (self.Y - Y_pred))

            m -= self.learning_rate * m_gradient
            b -= self.learning_rate * b_gradient

            # Вычисляем среднеквадратичную ошибку (MSE) на каждой итерации
            mse = np.mean((self.Y - Y_pred) ** 2)
            errors.append(mse)

        self.m, self.b = m, b
        print(f"Optimized parameters: m = {self.m}, b = {self.b}")
        print(f"Final error (MSE): {errors[-1]}")

    def predict(self):
        return self.m * self.X ** 2 - self.b * self.X ** 3

    def test_pipeline(self, test_number: int = 0):
        print(f"{i}. Testing learning rate = {learning_rate} with iterations = {iterations}")
        start_time = time.time()

        self.GenerateData()
        self.fit()
        self.PlotData(f"result_{test_number}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for {i}: {elapsed_time:.4f} seconds\n")

if __name__ == '__main__':
    sys.stdout = open('result.txt', 'w')
    test_params = [
        (1e-05, 1e3),
        (1e-08, 1e6), # too much (~10 sec)
        (1e-08, 1e5), # too much mse
        (1e-06, 1e5)
    ]

    for i in range(len(test_params)):
        learning_rate, iterations = test_params[i]
        iterations = int(iterations)
        GradientDescentLinearRegression(
            learning_rate,
            iterations
        ).test_pipeline(i)

    sys.stdout.close()
