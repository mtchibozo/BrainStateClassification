from model.training import train_model
from model.evaluation import (evaluate_proposed_model, evaluate_svc, evaluate_linear_svc, evaluate_random_forest,
                              evaluate_hanson_neural_net, compute_roc)
from metrics.plots import plot_validation_result, plot_roc

# Set path to data directory in the following format, data_path = 'path/to/data/directory'

# For example, Author used MVPA_hyperalignment.ipynb to generate `X_hyp_v2.csv` and `Y_hyp_v2.csv` and saved two csv
# files in data_path

# # Cell below applies to the author only.
#data_path = '/Users/gimdong-geon/Documents/GitHub/BrainStateClassification/data'
data_path = '.\data'


# # Training Model
# # Actual experiment used epochs equal to 5000
train_model(data_path, epochs=1)


# # Evaluate Proposed Model and save evaluation results
evaluate_proposed_model(data_path)

# # Evaluate Alternative Models using the same data and print results
evaluate_svc(data_path)
evaluate_linear_svc(data_path)
evaluate_random_forest(data_path)
evaluate_hanson_neural_net(data_path)

# Plot validation accuracy and loss plots. Plots are saved at `data_path`
plot_validation_result(data_path)

# Plot ROC curve. Plots are saved at `data_path`
plot_roc(data_path, holdout_index=10)
