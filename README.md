# Sample workflow for Vertex AI and Kubeflow Pipelines
## Set up python env
```sh
pip install poetry
pip install flake8 mypy  # assume to use it for coding conventions
poetry install
```

## Run on local Kubeflow Pipelines
### Set up Kubeflow Pipelines
1. [Deploy Kubeflow Pipelines to local](https://www.kubeflow.org/docs/components/pipelines/installation/localcluster-deployment/)
2. Run `nohup kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &` for external scripts to to access to KFP

### Run pipeline
```sh
echo "KFP_LOCALHOST=http://localhost:8080" >> .env
poetry run python src/main.py
```

## Run on Vertex AI Pipelines
1. Set up account `gcloud auth application-default login`
2. Check values in `.env.default`. To override them, add lines to `.env`
3. run `poetry run python src/main.py`

### request to endpoint
`[[5.1, 3.5, 1.4, 0.2]]` is input to predict function of scikit-learn modesl

```sh
ENDPOINT_ID="..."
PROJECT_ID="..."

curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://asia-northeast1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/asia-northeast1/endpoints/${ENDPOINT_ID}:predict \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```
