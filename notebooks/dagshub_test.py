import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/Poojarai24/water-potability-prediction.mlflow")

dagshub.init(repo_owner='Poojarai24', repo_name='water-potability-prediction', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)