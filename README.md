# advanced_mlops_project
This is the study project on HSE master program - Advanced MLOps.
Here some Ops tools are set up: poetry, pre-commit, dvc, MLFlow, hydra.

The main goal was to try to wrap a machine learning project (simple neural network) into a python package with useful ML development tools.

The tasks that solves this repository: classify cat and dogs by photo, construct consistent and right ML repository.

### To run the project you need:
1) Clone repository
2) In the working directory install all needed dependencies __```poetry install```__
3) Run __```dvc pull```__ to download all needed data for training and pre-trained model
4) Run __``` mlflow server --backend-store-uri file:///your/user/path/to/repo/advanced_mlops_project/advanced_mlops_project/mlflow_runs --default-artifact-root file:///your/user/path/to/repo/advanced_mlops_project/mlruns```__ to start local MLFlow
4) File __```advanced_mlops_project/train.py```__ starts training of the neural network, saves the model, log metrics in MLFlow
5) File __```advanced_mlops_project/infer.py```__ starts the inference of the best saved model on the test data and saves results


\* To contribute into project install dev package  __```pre-commit install```__
