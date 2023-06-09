# IrisDatasetGridSearchCV

Just learned about how GridSearchCV can be used for hyperparamater tuning so I gave it a go with Iris dataset.

I only used 2 feauteres for this along with its corresponding labels. 

I had to make a deep copy of the model in order to pass it in the GridSearchCV.

After the GridSearchCV was done, best model was at: epochs=250, input_shape=2, lr=0.1, output_shape=2) with a best score of 97%.
