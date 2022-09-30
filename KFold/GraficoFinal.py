
import matplotlib.pyplot as plt

left_coordinates=[1,2,3,4]
heights=[87.59,77.02,85.84,63.33]
bar_labels=['Random Forest','Naive Bayes','SVM','KNN']
plt.bar(left_coordinates,heights,tick_label=bar_labels,width=0.5,color=['red','black'])
plt.title("Acurácia")
plt.show()