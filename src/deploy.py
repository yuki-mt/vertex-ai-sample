from kfp.v2.dsl import component, Artifact, Input, Model, Output


@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9",
)
def deploy_model(
    model: Input[Model],
    project: str,
    region: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    deployed_model = aiplatform.Model.upload(
        display_name="sample-model-pipeline",
        artifact_uri=model.uri.replace("model", ""),
        # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
        serving_container_image_uri=("asia-docker.pkg.dev/vertex-ai/"
                                     "prediction/sklearn-cpu.0-24:latest")
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")
    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name
