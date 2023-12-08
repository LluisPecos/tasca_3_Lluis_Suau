# Preparación de los datos

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:,[2,3]] # longitud y ancho pétalo
y = iris.target # 0, 1, 2 según si es setosa, versicolor o virginica

# Si no dispusiéramos del datasets.load_iris() de sklearn, obtendría X e y de la siguiente forma
# df = pd.read_csv('../datasets/iris.data', header=None)
# X = df.iloc[:,[2,3]]
# class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# y = df[4].map(class_mapping)

# Separación en conjuntos de entrenamiento (70%) y test (30%)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Estandarización

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)