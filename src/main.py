from kfp.v2.dsl import pipeline, Condition
from kfp.dsl import PipelineExecutionMode
from kfp.v2 import compiler
from kfp import Client
from google.cloud import aiplatform
import os
from dotenv import load_dotenv
from datetime import datetime


from dataset import get_dataframe
from train import sklearn_train
from deploy import deploy_model


def load_env():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    load_dotenv(os.path.join(base_dir, ".env.default"))
    load_dotenv(os.path.join(base_dir, ".env"), override=True)


@pipeline(name="sample-pipeline")
def define_pipeline(project: str, region: str, run_local: bool):
    dataset_task = get_dataframe().set_cpu_limit("1")\
        .set_memory_limit("2Gi")
    model_task = sklearn_train(dataset_task.output)
    # flake8: noqa
    with Condition(run_local == False):
        deploy_model(model=model_task.outputs["model"],
                     project=project,
                     region=region)


def main():
    load_env()
    if "KFP_LOCALHOST" in os.environ:
        client = Client(os.environ["KFP_LOCALHOST"])
        client.create_run_from_pipeline_func(
            define_pipeline,
            mode=PipelineExecutionMode.V2_COMPATIBLE,
            arguments={
                "project": "",
                "region": "",
                "run_local": True,
            }
        )
    else:
        pipeline_path = "pipeline_config/sample_pipeline.json"
        compiler.Compiler().compile(pipeline_func=define_pipeline,
                                    package_path=pipeline_path)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        aiplatform.PipelineJob(
            display_name="sample-pipeline",
            pipeline_root=os.environ["PROJECT_ROOT"],
            template_path=pipeline_path,
            job_id=f"sample-pipeline-small-{timestamp}",
            parameter_values={
                "project": os.environ["PROJECT_ID"],
                "region": os.environ["REGION"],
                "run_local": False,
            },
            enable_caching=True,
            location=os.environ["REGION"],
        ).submit()


if __name__ == '__main__':
    main()
