import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,  precision_score, f1_score
from random import randint
#Transformando as strings em int
ds = pd.read_csv (r'tentando.csv')
ds = ds.drop([82], axis = 0)
ds_geral = ds.iloc[:, 0:7].values
ds_geral[:, 5] = LabelEncoder().fit_transform(ds_geral[:, 5])
ds_geral[:, 6] = LabelEncoder().fit_transform(ds_geral[:, 6])
df_geral = pd.DataFrame(ds_geral)

##################################################################
previsores = df_geral.iloc[:, 0:6].values
previsoresdf = pd.DataFrame(previsores)


alvo = df_geral.iloc[:, 6].values
alvo = alvo.astype('int')
alvodf = pd.DataFrame(alvo)

###################################################################
"""
kf = KFold(n_splits=6, shuffle=True, random_state=8)
svm = SVC(kernel='rbf')
label = ['0', '1', '2', '3', '4', '5']
alvodf = alvodf.squeeze().ravel()
previsaofinal = cross_val_predict(svm, previsoresdf, alvo, cv = kf)
#metrics.f1_score(alvo, previsaofinal, average='weighted', labels=np.unique(alvo))
print(classification_report(alvo, previsaofinal))
print(confusion_matrix(alvo, previsaofinal))
"""









lista_fum = list()
lista_acuracia = list()
lista_precisao = list()
i=0
while i<30:
    x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, alvo, test_size = 0.3, random_state= randint(0,100))

 ###################################################################
    naive = GaussianNB()
    naive.fit(x_treino, y_treino)

    previsao_teste = naive.predict(x_teste)

    #print(accuracy_score(y_teste, previsao_teste)*100)

    previsao_treino = naive.predict(x_treino)
    acuracia = accuracy_score(y_treino, previsao_treino)
    #lista_resultado.append(acuracia)
    #print(confusion_matrix(y_treino, previsao_treino))
    fum = f1_score(y_teste, previsao_teste, average='micro')
    acuracia = accuracy_score(y_teste, previsao_teste)
    precisao = precision_score(y_treino, previsao_treino, average='micro')
    lista_fum.append(fum)
    lista_acuracia.append(acuracia)
    lista_precisao.append(precisao)
    i+= 1

media_fum = sum(lista_fum)/len(lista_fum)
media_acuracia = sum(lista_acuracia)/len(lista_acuracia)
media_precisao = sum(lista_precisao)/len(lista_precisao)
print('',media_fum,'\n',media_acuracia,'\n',media_precisao)
