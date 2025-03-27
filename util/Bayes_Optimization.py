import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

class Bayes_Optimization:
    def __init__(self):
        pass
    
    # 黑箱函数
    def black_box_function(self, x):
        '''
        拥有局部最优解
        '''
        return np.sin(3 * x) + x ** 2 + 0.7 * x
    
    # 期望改进采集函数
    def expected_improvement(self, X, model, f_best, xi=0.01):
        '''
        X: [num_sample, num_feature]
        model:高斯核
        f_best: 训练点中的最优值
        '''
        
        # 获取采样点的函数值的正态分布
        mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        
        # 根据分布打分
        Z = (mu - f_best - xi) / sigma
        EI = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return EI
    
    def bayesian_optimization(self, f, bounds, num_init=5, num_iter=20, xi=0.01, alpha=1-10):
        
        # 范围内随机采样X，用黑盒函数计算y
        X = np.random.uniform(bounds[0][0], bounds[0][1], num_init).reshape(-1, 1)
        y = f(X).reshape(-1, 1)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel, alpha=alpha, normalize_y=True)
        
        for epoch in range(num_iter):
        
            model.fit(X, y)
            
            def neg_ei(X):
                f_best = np.max(y)
                return (-1) * self.expected_improvement(X.reshape(-1, 1), model, f_best, xi)
            
            res = minimize(fun=neg_ei, x0=np.random.uniform(bounds[0][0], bounds[0][1]), bounds=bounds, method='L-BFGS-B')
            x_next = res.x
            y_next = f(x_next)
            
            X = np.vstack((X, x_next))
            y = np.vstack((y, y_next))
            print(len(X), len(y))
            
        return model, X, y
        
    def main(self):
        # Bounds格式要求，每一维的bound序列
        bounds = [(-5, 5)]
        model, X, y = self.bayesian_optimization(self.black_box_function, 
                                                  bounds, 
                                                  num_init=5,
                                                  num_iter=30,
                                                  xi=0.01,
                                                  alpha=1e-3)
        
        x_plot = np.arange(bounds[0][0], bounds[0][1], 0.001).reshape(-1, 1)
        y_true = self.black_box_function(x_plot)
        y_pred, sigma = model.predict(x_plot, return_std=True)
        
        plt.plot(x_plot, y_true, 'r--', label='True_function')
        plt.plot(x_plot, y_pred, 'b-', label='GP_prediction')
        plt.fill_between(x_plot.reshape(-1,), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue')
        plt.scatter(X, y, c='red', s=50, zorder=10, label='Samples')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.savefig('../logs/Bayes_opt.png')
        
optimizer = Bayes_Optimization()
optimizer.main()