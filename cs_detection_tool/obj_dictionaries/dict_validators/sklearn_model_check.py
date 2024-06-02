from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), DecisionTreeRegressor(),
          RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), SVR(),
          KNeighborsRegressor(), MLPRegressor(), GaussianProcessRegressor(), GaussianNB(),
          LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), KMeans(),
          AgglomerativeClustering(), DBSCAN(), SpectralClustering()]

for model in models:
    print(isinstance(model, BaseEstimator))