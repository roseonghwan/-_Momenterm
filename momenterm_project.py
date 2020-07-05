import numpy as np
import matplotlib.pyplot as plt

class Momenterm:

    # 데이터 추출
    def setData(self, X_min, X_max, X, Y):
        plt.figure(figsize=(8, 5))
        plt.plot(X, Y, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
        plt.xlim(X_min, X_max)
        plt.xlabel('Study Time(hour)')
        plt.ylabel('Score')
        plt.grid(True)
        plt.show()

    # 평균 제곱 오차의 기울기
    def dmse_line(self, x, t, w):
        y = w[0] * x + w[1]
        d_w0 = 2 * np.mean((y - t) * x)
        d_w1 = 2 * np.mean(y - t)
        return d_w0, d_w1

    # 모멘텀 구현
    def fit_line_num(self, x, t):
        momentermTest = Momenterm()
        w_init = [5.0,85.0]
        alpha=0.0015
        mu = 0.9
        i_max = 1000000
        eps = 0.01
        w_i = np.zeros([i_max, 2])
        v_i = np.zeros([i_max, 2])
        w_i[0, :] = w_init
        v_i[0, :] = [0, 0]
        for i in range(1, i_max):
            dmse = momentermTest.dmse_line(x, t, w_i[i - 1])
            v_i[i, 0] = mu * v_i[i - 1, 0] - alpha * dmse[0]
            v_i[i, 1] = mu * v_i[i - 1, 1] - alpha * dmse[1]
            w_i[i + 1, 0] = w_i[i, 0] + v_i[i, 0]
            w_i[i + 1, 1] = w_i[i, 1] + v_i[i, 1]
            if max(np.absolute(dmse)) < eps:
                break
        w0 = w_i[i, 0]
        w1 = w_i[i, 1]
        return w0, w1

    # 그래프 긋기
    def drawGraph(self, X, Y, X_min, X_max, W0, W1):
        plt.figure(figsize=(8,5))
        plt.plot(X, Y, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
        plt.plot(X, W0 * X + W1)
        plt.xlim(X_min, X_max)
        plt.xlabel('Study Time(hour)')
        plt.ylabel('Score')
        plt.grid(True)
        plt.show()

    # 로젠브록 함수
    def f2(self, x, y):
        return (1 - x)**2 + 100.0 * (y - x**2)**2

    # 로젠브록 도함수
    def f2g(self, x, y):
        return np.array((2.0 * (x - 1) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)))

    # 로젠브록 함수 구현
    def rosenbrock(self, X, Y):
        momentermTest = Momenterm()
        alpha = 0.0001
        momenterm = 0.9
        x, y = -1, -1
        vx = 0
        vy = 0
        for i in range(35):
            g = momentermTest.f2g(x, y)
            plt.scatter(x, y)
            vx = momenterm * vx - alpha * g[0]
            vy = momenterm * vy - alpha * g[1]
            x = x + vx
            y = y + vy
        xx = np.linspace(-4, 4, 800)
        yy = np.linspace(-3, 3, 600)
        X, Y = np.meshgrid(xx, yy)
        Z = momentermTest.f2(X, Y)

        levels = np.logspace(-1, 3, 10)
        plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
        plt.contour(X, Y, Z, colors="gray",
                    levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000])
        plt.plot(1, 1, 'ro', markersize=10)

        plt.xlim(-4, 4)
        plt.ylim(-3, 3)
        plt.xticks(np.linspace(-4, 4, 9))
        plt.yticks(np.linspace(-3, 3, 7))
        plt.show()

if __name__ == '__main__':
    momentermTest = Momenterm()
    np.random.seed(seed=1)
    X_min = 1
    X_max = 8
    X_n = 50
    X = 1 + 7 * np.random.rand(X_n)
    Y = 2.5 * X + 70 + 10 * np.random.rand(X_n)
    momentermTest.setData(X_min, X_max, X, Y)
    W0, W1 = momentermTest.fit_line_num(X, Y)
    momentermTest.drawGraph(X, Y, X_min, X_max, W0, W1)
    momentermTest.rosenbrock(X, Y)
