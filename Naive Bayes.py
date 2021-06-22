"""
Imports necessários
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#nome das colunas da tabela
col_names = ['id', 'partner', 'age', 'age_o', 'goal', 'date', 'go_out', 'int_corr', 'length', 'met', 'like', 'prob', 'match']

#leitura do dataset
dataset = pd.read_csv("speedDating_trab.csv", header=0, names=col_names)

#pre-processamento
#primeiro descartar todas as linhas com ocorrências de NA
dataset = dataset.dropna()

#depois descartar os atributos desnecessários
dataset = dataset.drop('id', axis=1)
dataset = dataset.drop('partner', axis=1)

#definir o tamanho do conjunto de treino (80% do dataset original)
train_length = int(len(dataset) * 0.8)

"""
Dividir o dataset em conjuntos de treino e teste com a percentagem acima definida
"""
y = dataset['match']
X = dataset.drop('match', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

"""
Treinar o modelo
"""
gnb = GaussianNB().fit(X_train, y_train)

"""
Fazer previsão dos resultados
"""
pred_y = gnb.predict(X_test)

"""
Verificar a precisão da previsão
"""
print("Precisão da previsão do Gaussian Naive Bayes: ", accuracy_score(y_test, pred_y)*100, "%")
