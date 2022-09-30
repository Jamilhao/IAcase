
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn import metrics
from random import randint
#Transformando as strings em int
ds = pd.read_csv (r'tentando.csv')
ds = ds.drop([82], axis = 0)
ds_geral = ds.iloc[:, 0:7].values
ds_geral[:, 5] = LabelEncoder().fit_transform(ds_geral[:, 5])
ds_geral[:, 6] = LabelEncoder().fit_transform(ds_geral[:, 6])
df_geral = pd.DataFrame(ds_geral)
#print(df_geral)

##################################################################
previsores = df_geral.iloc[:, 0:6].values
previsoresdf = pd.DataFrame(previsores)

alvo = df_geral[6].to_list()
alvo = list(map(int, alvo))
alvodf = pd.DataFrame(alvo)


i = 0

lista_fum = list()
lista_acuracia = list()
lista_precisao = list()
while i<30:
    kf = KFold(n_splits=6, shuffle=True, random_state=randint(0, 100))    #Efetuar teste final com 30 random_state's diferentes
    for treino_i, teste_i in kf.split(df_geral):
         x_treino, x_teste = previsoresdf.iloc[treino_i], previsoresdf.iloc[teste_i]
         y_treino, y_teste = alvodf.iloc[treino_i], alvodf.iloc[teste_i]
         svm = SVC(kernel='rbf')
         y_treino = y_treino.squeeze().ravel()
         svm.fit(x_treino, y_treino)
         previsao_teste = svm.predict(x_teste)
         previsao_treino = svm.predict(x_treino)
         fum = f1_score(y_teste, previsao_teste, average='micro')
         acuracia = accuracy_score(y_teste, previsao_teste)
         precisao = precision_score(y_treino, previsao_treino, average='micro')
         lista_fum.append(fum)
         lista_acuracia.append(acuracia)
         lista_precisao.append(precisao)
         i += 1

media_fum = sum(lista_fum) / len(lista_fum)
media_acuracia = sum(lista_acuracia) / len(lista_acuracia)
media_precisao = sum(lista_precisao) / len(lista_precisao)
print('', media_fum, '\n', media_acuracia, '\n', media_precisao)
"""
svm = SVC(kernel='rbf')
label = ['0', '1', '2', '3', '4', '5']
alvodf = alvodf.squeeze().ravel()
previsaofinal = cross_val_predict(svm, previsoresdf, alvo, cv = kf)
metrics.f1_score(alvo, previsaofinal, average='weighted', labels=np.unique(alvo))
#print(classification_report(alvo, previsaofinal))
#matriz = confusion_matrix(alvo, previsaofinal, labels=label)
#print(matriz)
"""








#x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, alvo, test_size = 0.3, random_state=0)
##################################################################

""""
previsao_teste = svm.predict(x_teste)
print(accuracy_score(y_teste, previsao_teste))
print(confusion_matrix(y_teste, previsao_teste))
print(classification_report(y_teste, previsao_teste))
previsao_treino = svm.predict(x_treino)
print(accuracy_score(y_treino, previsao_treino))
print(confusion_matrix(y_treino, previsao_treino))
print(classification_report(y_treino, previsao_treino))
"""