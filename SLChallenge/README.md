# SLChallenge

This is a brief description for each of the files used
to generate the data required by the challenge.

There are 4 main files:

### generador_de_embeddings_de_entrenamiento.py
Creates the training embeddings by using the training data 
(hsc_lense, and hsc_non-lense)


### generador_de_embeddings_reales_y_CSV_para_challenge
Creates the test embeddings using the training embeddings, 
and generates the classification and regression data, plus the CSV file for the challenge.
It also generates a CSV file to track failed files and reason, plus a TXT file
with the ids of the failed files.

### red_neuronal_entrenamiento.py
Creates the training NN with the training enbeddings and the CSV file with
the classes (1 - lense, 0 - non-lense)

### red_neuronal.py
Creates the NN by using the test embeddings and also produces the CSV file for 
the challenge and the weights (best_lens_classifier.pth)

