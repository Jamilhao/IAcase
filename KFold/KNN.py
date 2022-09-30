import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from random import randint

ds = pd.read_csv (r'tentando.csv')
ds = ds.drop([82], axis = 0) #tirei a classe com uma estrela apenas
ds_geral = ds.iloc[:, 0:7].values #peguei apenas as colunas que eu vou usar
ds_geral[:, 5] = LabelEncoder().fit_transform(ds_geral[:, 5])#Transformei variaveis string em int
ds_geral[:, 6] = LabelEncoder().fit_transform(ds_geral[:, 6])
df_geral = pd.DataFrame(ds_geral)#passei o dataframe pra dataset
previsores = df_geral.iloc[:, 0:6].values
previsoresdf = pd.DataFrame(previsores)

alvo = df_geral[6].to_list()
alvo = list(map(int, alvo))
alvodf = pd.DataFrame(alvo)
lista_fum = list()
lista_acuracia = list()
lista_precisao = list()
i=0

x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, alvo, test_size = 0.3, random_state=0)
while i<30:
    knn = KNeighborsClassifier(n_neighbors=randint(1, 100), metric='minkowski', p=2)
    knn.fit(x_treino, y_treino)

    previsoes_teste = knn.predict(x_teste)
    resultado = accuracy_score(y_teste, previsoes_teste)
    previsoes_treino = knn.predict(x_treino)
    fum = f1_score(y_teste, previsoes_teste, average='micro')
    acuracia = accuracy_score(y_teste, previsoes_teste)
    precisao = precision_score(y_treino, previsoes_treino, average='micro')
    lista_fum.append(fum)
    lista_acuracia.append(acuracia)
    lista_precisao.append(precisao)
    i += 1

media_fum = sum(lista_fum) / len(lista_fum)
media_acuracia = sum(lista_acuracia) / len(lista_acuracia)
media_precisao = sum(lista_precisao) / len(lista_precisao)
print('', media_fum, '\n', media_acuracia, '\n', media_precisao)

