# PAI_Pr3_BayesianOptimization
Project 3 for the course of Probabilistic Artificial Intelligence. Used Bayesian optimization to tune one hyperparameter of a machine learning model subject to a constraint on a property of the model.
Let θ∈Θ denote the hyperparameter of interest, e.g., the number of layers in a deep neural network. The objective is to find a network that makes accurate and fast predictions. To this end, the model can be trained for a specific value of θ on a training data set. From this,a corresponding accuracy on the validation data set and an average prediction speed can be optained. In this context, the goal is to find the value of the hyperparameter that induces the highest possible validation accuracy, such that a requirement on the average prediction speed is satisfied.

# HOW TO EXECUTE
1. Download Docker
2. To run from Linux
  $ bash runner.sh

# SHORT REPORT
To model the validation accuracy, a GaussianProccessRegressor with Matern Kernel (length_scale=0.5, nu=2.5, var=0.5) was used.
To model the prediction speed, a GaussianProccessRegressor with Matern Kernel (length_scale=0.5, nu=2.5, var=1.5, mean=1.5) was used.
The next_reccomandation is implemented by optimizing the acquisition function (optimize_acquisition_function).
The actual acquisition function is implemented by using the estimation of validation accuracy scaled by the likelihood of the prediction speed (which is a gaussian distribution by hypothesis).
The add_data_point function simply refits the GaussianProcessRegressor models on the whole training dataset obtained putting together the old dataset D and the new datapoint x.
Finally, the get_solution returns the solution such that it maximizes the validation accuracy over all the possible solutions whose prediction speed is above the  threshold k=1.2.

