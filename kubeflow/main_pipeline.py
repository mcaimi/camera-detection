# Import KubeFlow Pipelines library
import kfp

# Import objects from the DSL library
from kfp.dsl import pipeline
from kfp import kubernetes

# Component imports
from fetch_data import fetch_data, fetch_model
from persistent_data import unzip_dataset, unzip_model
from train_model import train_model


# Pipeline definition

# name of the data connection that points to the s3 model storage bucket
artifacts_connection_secret_name = 's3-artifacts'
data_connection_secret_name = 's3-models'
kaggle_api_secrets = 'yolo-kaggle'
huggingface_api_secret = 'huggingface-secret'


# Create pipeline
@pipeline(
  name='yolo-custom-training-pipeline',
  description='Dense Neural Network CNN Image Detector based on YOLO'
)
def training_pipeline(hyperparameters: dict,
                      model_name: str,
                      dataset_name: str,
                      version: str,
                      s3_deployment_name: str,
                      s3_region: str,
                      author_name: str,
                      cluster_domain: str,
                      huggingface_repo: str,
                      data_mount_path: str,
                      prod_flag: bool):

    # Fetch Data from Kaggle
    fetch_dataset_task = fetch_data(dataset_name=dataset_name, version=version)
    kubernetes.use_secret_as_env(
        fetch_dataset_task,
        secret_name=huggingface_api_secret,
        secret_key_to_env={
            'HF_HOME': 'HF_HOME',
            'HF_TOKEN': 'HF_TOKEN',
        },
    )

    # download base model from HF
    fetch_model_task = fetch_model(model_name=huggingface_repo,
                                   hyperparameters=hyperparameters)
    kubernetes.use_secret_as_env(
        fetch_model_task,
        secret_name=kaggle_api_secrets,
        secret_key_to_env={
            'KAGGLE_USERNAME': 'KAGGLE_USERNAME',
            'KAGGLE_KEY': 'KAGGLE_KEY',
            'KAGGLEHUB_CACHE': 'KAGGLEHUB_CACHE',
        },
    )

    # prepare data on persistent storage for training
    #data_disk = kubernetes.CreatePVC(
    #    pvc_name_suffix="-training-pvc",
    #    access_modes=['ReadWriteOnce'],
    #    size='50Gi',
    #    storage_class_name='ocs-storagecluster-ceph-rbd',
    #)

    # unzip dataset to persistent storage
    unzip_dataset_task = unzip_dataset(data_dir=data_mount_path,
                                       dataset=fetch_dataset_task.outputs["dataset"])
    unzip_dataset_task.after(fetch_dataset_task)

    # mount persistent volume...
    kubernetes.mount_pvc(
        unzip_dataset_task,
        pvc_name='training', #data_disk.outputs['name'],
        mount_path="/data",
    )

    # unzip model to persistent storage
    unzip_model_task = unzip_model(data_dir=data_mount_path,
                                   model=fetch_model_task.outputs["original_model"])
    unzip_model_task.after(unzip_dataset_task)

    # mount persistent volume...
    kubernetes.mount_pvc(
        unzip_model_task,
        pvc_name='training',
        mount_path="/data",
    )

    # train model
    train_model_task = train_model(original_model=fetch_model_task.outputs["original_model"],
                                   dataset=fetch_dataset_task.outputs["dataset"],
                                   hyperparameters=hyperparameters,
                                   data_mount_path=data_mount_path)
    train_model_task.set_cpu_limit("8")
    train_model_task.set_memory_limit("24G")
    # mount persistent volume...
    kubernetes.mount_pvc(
        train_model_task,
        pvc_name='training',
        mount_path="/data",
    )
    train_model_task.after(unzip_model_task)
    train_model_task.after(unzip_dataset_task)

    # remove pvc after pipeline completes
    #delete_datadisk = kubernetes.DeletePVC(
    #    pvc_name=data_disk.outputs['name']
    #).after(unzip_model_task)


# start pipeline
if __name__ == '__main__':
    metadata = {
        "hyperparameters": {
            "epochs": 1,
            "batch": 2,
            "img_size": 640,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "job": "detect",
            "run_name": "train",
            "checkpoint": "yolo11x.pt",
            "optimizer": "AdamW",
            "augment": True,
            "training_job_descriptor": "military/aircraft_names.yaml",
        },
        "model_name": "yolo-custom-finetuned",
        "dataset_name": "rookieengg/military-aircraft-detection-dataset-yolo-format",
        "version": "1",
        "s3_deployment_name": "minio-s3-s3-minio-dev",
        "s3_region": "us",
        "author_name": "DevOps Team",
        "cluster_domain": "apps.xxx-yyy.local",
        "huggingface_repo": "Ultralytics/YOLO11",
        "data_mount_path": "/data",
        "prod_flag": False
    }

    namespace_file_path =\
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint =\
        f'https://ds-pipeline-dspa.{namespace}.svc:8443'

    sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(sa_token_file_path, 'r') as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert =\
        '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'

    # compile pipeline for debugging
    from kfp import compiler
    compiler.Compiler().compile(training_pipeline, 'pipeline.yaml')

    # Run pipeline on cluster
    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert
    )

    client.create_run_from_pipeline_func(
        training_pipeline,
        arguments=metadata,
        experiment_name="yolo-custom-training-pipeline",
        enable_caching=True
    )