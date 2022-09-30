import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from random import randint


left_coordinates=[1,2,3,4]
heights=[87.59,77.02,85.84,63.33]
bar_labels=['Random Forest','Naive Bayes','SVM','KNN']
plt.bar(left_coordinates,heights,tick_label=bar_labels,width=0.5,color=['red','black'])
plt.title("Acurácia")
plt.show()


ds = pd.read_csv (r'tentando.csv')
ds = ds.drop([82], axis = 0)
ds_geral = ds.iloc[:, 0:7].values
ds_geral[:, 5] = LabelEncoder().fit_transform(ds_geral[:, 5])
ds_geral[:, 6] = LabelEncoder().fit_transform(ds_geral[:, 6])
df_geral = pd.DataFrame(ds_geral)
previsores = df_geral.iloc[:, 0:6].values
previsoresdf = pd.DataFrame(previsores)

alvo = df_geral[6].to_list()
alvo = list(map(int, alvo))
alvodf = pd.DataFrame(alvo)

lista = list()
lista2 = list()
i=0
x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, alvo, test_size = 0.3, random_state=0)
while i<30:
    random = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth= 3, random_state= randint(0,100))
    random.fit(x_treino, y_treino)
    previsoes_teste = random.predict(x_teste)
    resultado = accuracy_score(y_teste, previsoes_teste)
    lista.append(resultado)
    previsoes_treino = random.predict(x_treino)
    resultado2 = accuracy_score(y_treino, previsoes_treino)
    lista2.append(resultado2)
    i +=1

random_forest = (sum(lista)/len(lista))*100
print(random_forest)


lista_resultado = list()
i=0
while i<30:
    x_treino, x_teste, y_treino, y_teste = train_test_split(previsores, alvo, test_size = 0.3, random_state= randint(0,100))

    ###################################################################
    naive = GaussianNB()
    naive.fit(x_treino, y_treino)

    previsao_teste = naive.predict(x_teste)

    #print(accuracy_score(y_teste, previsao_teste)*100)

    previsao_treino = naive.predict(x_treino)
    acuracia = accuracy_score(y_treino, previsao_treino)*100
    lista_resultado.append(acuracia)
    #print(confusion_matrix(y_treino, previsao_treino))
    #print(classification_report(y_treino, previsao_treino))
    i+= 1
naive_bayes = sum(lista_resultado)/len(lista_resultado)

print(naive_bayes)

i = 0

scoresSVM = []
treinoSVM= []
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
         scoresSVM.append(accuracy_score(y_teste, previsao_teste))
         treinoSVM.append(accuracy_score(y_treino, previsao_treino))
         i += 1
SVM = (sum(scoresSVM)/len(scoresSVM))*100
#print(len(scoresSVM))
print(SVM)


lista1 = list()
i=0
while i<30:
    knn = KNeighborsClassifier(n_neighbors=randint(1, 100), metric='minkowski', p=2)
    knn.fit(x_treino, y_treino)

    previsoes_teste = knn.predict(x_teste)
    resultado = accuracy_score(y_teste, previsoes_teste)

    lista1.append(resultado)
    previsoes_treino = knn.predict(x_treino)
    resultado2 = accuracy_score(y_treino, previsoes_treino)

    i +=1
KNN = (sum(lista1)/len(lista1))*100
print(KNN)



