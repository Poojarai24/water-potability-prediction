import json
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Initialize DagsHub
dagshub.init(
    repo_owner="Poojarai24",
    repo_name="water-potability-prediction",
    mlflow=True
)

# Set tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Poojarai24/water-potability-prediction.mlflow"
)

# Load run info
with open("reports/run_info.json", "r") as f:
    run_info = json.load(f)

run_id = run_info["run_id"]
model_name = run_info["model_name"]

print("Registering model from Run ID:", run_id)

client = MlflowClient()

# 🔥 STEP 1 — Get experiment ID from run
run = client.get_run(run_id)
experiment_id = run.info.experiment_id

# 🔥 STEP 2 — Get logged model from this run
logged_models = client.search_logged_models(
    experiment_ids=[experiment_id],
    filter_string=f"source_run_id = '{run_id}'"
)

if not logged_models:
    raise Exception("No logged model found for this run.")

logged_model = logged_models[0]

print("Found logged model:", logged_model.name)
print("Model URI:", logged_model.model_uri)

# 🔥 STEP 3 — Create registered model if not exists
try:
    client.create_registered_model(model_name)
    print(f"Registered model '{model_name}' created.")
except Exception:
    print(f"Registered model '{model_name}' already exists.")

# 🔥 STEP 4 — Create model version using logged model URI
model_version = client.create_model_version(
    name=model_name,
    source=logged_model.model_uri,
    run_id=run_id
)

print(f"Model version created: {model_version.version}")

# 🔥 STEP 5 — Move to Staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging",
    archive_existing_versions=True
)

print(f"Model '{model_name}' version {model_version.version} moved to Staging.")