# PCA Algorithm in Classification of Multidimensional Data
This repository contains research on dimensionality reduciton algorithms for the classification task.
Chosen methods are: PCA, KPCA, SPCA, LDA and own implementation of L1PCA*.
They were compared on 5 different datasets and 4 popular classifiers (kNN, GNB, SVM and CART).
The goal is to find the best algorithm for handling dimensionality reduction in classification tasks.

# Results
The experiment was carried out on both synthetic and real data. 
The experiment protocol as well as results for real data can be found in src/pca_classification_real.ipynb file.
Based on statystical analysis (t-student test) LDA based algorithms were significantly better than algorithms based on PCA on the given datasets.
It is worth nothing that the conducted experiment is a small-scale trial. A good practice would be to repeat it using a larger range of datasets.
