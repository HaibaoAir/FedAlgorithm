import numpy as np
from sklearn.decomposition import PCA

data = np.array([[-1,-1,0,2,1],[2,0,0,-1,-1],[2,0,1,1,0]], dtype=np.float)
pca = PCA(n_components=2)
new_data = pca.fit