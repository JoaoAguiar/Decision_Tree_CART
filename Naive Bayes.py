import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Nome das colunas da tabela
cn = ['id', 'partner', 'age', 'age_o', 'goal', 'date', 'go_out', 'int_corr', 'length', 'met', 'like', 'prob', 'match']

dataset = pd.read_csv("speedDating_trab.csv", header=0, names=cn)

# Pre-processamento
# Remover linhas com ocorrências de NA e atributos desnecessários (id e partner)
dataset = dataset.dropna()
dataset = dataset.drop('id', axis=1)
dataset = dataset.drop('partner', axis=1)

# Tamanho do conjunto de treino (80% do dataset original)
train_length = int(len(dataset) * 0.8)

# Dividir o dataset em treino e teste
y = dataset['match']
X = dataset.drop('match', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

# Treinar o modelo
gnb = GaussianNB().fit(X_train, y_train)
# Prever o match
prediction = gnb.predict(X_test)

print("Precisão da previsão do Gaussian Naive Bayes: ", accuracy_score(y_test, prediction)*100, "%")
