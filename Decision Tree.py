"""
Imports necessários
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

#nome das colunas da tabela
col_names = ['id', 'partner', 'age', 'age_o', 'goal', 'date', 'go_out', 'int_corr', 'length', 'met', 'like', 'prob', 'match']

#nome das classe = nome das colunas
cn = col_names

#nome das features (sem o id, o partner, o match e o int_corr)
fn = ['age', 'age_o', 'goal','int_corr' ,'date', 'go_out', 'length', 'met', 'like', 'prob']

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
train_features = dataset.iloc[:train_length,:-1]
test_features = dataset.iloc[train_length:,:-1]
train_targets = dataset.iloc[:train_length,-1]
test_targets = dataset.iloc[train_length:,-1]


"""
Treinar o modelo
"""
clf = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)


"""
Prever o match com base nas restantes classes
"""
prediction = clf.predict(test_features)


"""
Verificar a precisão
"""
text_representation = tree.export_text(clf)

print(text_representation)
print()
print("Precisão da previsão: ",clf.score(test_features,test_targets)*100,"%")

"""
Gerar o ficheiro que após compilado gera a imagem da árvore
"""
dot_data = tree.export_graphviz(clf, out_file="tree.dot",  feature_names=fn, class_names=cn, filled=True)
