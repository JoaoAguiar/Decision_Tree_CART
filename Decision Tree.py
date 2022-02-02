import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

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
train_features = dataset.iloc[:train_length, :-1]
test_features = dataset.iloc[train_length:, :-1]
train_targets = dataset.iloc[:train_length, -1]
test_targets = dataset.iloc[train_length:, -1]

# Treinar o modelo
clf = DecisionTreeClassifier(criterion = 'entropy').fit(train_features, train_targets)
# Prever o match
prediction = clf.predict(test_features)
text_representation = tree.export_text(clf)

print(text_representation)
print()
print("Precisão da previsão: ", clf.score(test_features,test_targets)*100, "%")