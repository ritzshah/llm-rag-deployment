"""Compile KFP v2 pipelines for RHOAI 3.x Data Science Pipelines."""
from kfp import dsl, compiler


RUNTIME_IMAGE = "quay.io/modh/runtime-images@sha256:7dd23e58291cad7a0ab4a8e04bda06492f2c027eb33b226358380db58dcdd60b"


@dsl.component(base_image=RUNTIME_IMAGE)
def data_ingestion(
    minio_endpoint: str = "http://minio.ic-shared-rag-minio.svc:9000",
    minio_bucket: str = "pipeline",
    cos_directory: str = "pipeline-code",
):
    """Run the Langchain-PgVector-Ingest notebook via Elyra bootstrapper."""
    import subprocess, os
    os.environ["HOME"] = "/tmp"
    os.makedirs("/tmp/jupyter-work-dir", exist_ok=True)
    os.chdir("/tmp/jupyter-work-dir")

    # Download bootstrapper
    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/bootstrapper.py",
        "--output", "bootstrapper.py"
    ], check=True)
    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/requirements-elyra.txt",
        "--output", "requirements-elyra.txt"
    ], check=True)

    subprocess.run(["python3", "-m", "pip", "install", "packaging"], check=True)
    subprocess.run(["python3", "-m", "pip", "freeze"], stdout=open("requirements-current.txt", "w"))
    subprocess.run([
        "python3", "bootstrapper.py",
        "--pipeline-name", "data_ingestion_pipeline",
        "--cos-endpoint", minio_endpoint,
        "--cos-bucket", minio_bucket,
        "--cos-directory", cos_directory,
        "--cos-dependencies-archive", "Langchain-PgVector-Ingest.tar.gz",
        "--file", "llm-rag-deployment/examples/pipelines/Langchain-PgVector-Ingest.ipynb",
    ], check=True, env={
        **os.environ,
        "ELYRA_RUNTIME_ENV": "kfp",
        "ELYRA_ENABLE_PIPELINE_INFO": "True",
        "ELYRA_WRITABLE_CONTAINER_DIR": "/tmp",
    })


@dsl.component(base_image=RUNTIME_IMAGE)
def data_query(
    minio_endpoint: str = "http://minio.ic-shared-rag-minio.svc:9000",
    minio_bucket: str = "pipeline",
    cos_directory: str = "pipeline-code",
):
    """Run the Langchain-PgVector-Query notebook via Elyra bootstrapper."""
    import subprocess, os
    os.environ["HOME"] = "/tmp"
    os.makedirs("/tmp/jupyter-work-dir", exist_ok=True)
    os.chdir("/tmp/jupyter-work-dir")

    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/bootstrapper.py",
        "--output", "bootstrapper.py"
    ], check=True)
    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/requirements-elyra.txt",
        "--output", "requirements-elyra.txt"
    ], check=True)

    subprocess.run(["python3", "-m", "pip", "install", "packaging"], check=True)
    subprocess.run(["python3", "-m", "pip", "freeze"], stdout=open("requirements-current.txt", "w"))
    subprocess.run([
        "python3", "bootstrapper.py",
        "--pipeline-name", "data_ingestion_pipeline",
        "--cos-endpoint", minio_endpoint,
        "--cos-bucket", minio_bucket,
        "--cos-directory", cos_directory,
        "--cos-dependencies-archive", "Langchain-PgVector-Query.tar.gz",
        "--file", "llm-rag-deployment/examples/pipelines/Langchain-PgVector-Query.ipynb",
    ], check=True, env={
        **os.environ,
        "ELYRA_RUNTIME_ENV": "kfp",
        "ELYRA_ENABLE_PIPELINE_INFO": "True",
        "ELYRA_WRITABLE_CONTAINER_DIR": "/tmp",
    })


@dsl.component(base_image=RUNTIME_IMAGE)
def test_response_time(
    minio_endpoint: str = "http://minio.ic-shared-rag-minio.svc:9000",
    minio_bucket: str = "pipeline",
    cos_directory: str = "pipeline-code",
):
    """Run the response time test."""
    import subprocess, os
    os.environ["HOME"] = "/tmp"
    os.makedirs("/tmp/jupyter-work-dir", exist_ok=True)
    os.chdir("/tmp/jupyter-work-dir")

    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/bootstrapper.py",
        "--output", "bootstrapper.py"
    ], check=True)
    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/requirements-elyra.txt",
        "--output", "requirements-elyra.txt"
    ], check=True)

    subprocess.run(["python3", "-m", "pip", "install", "packaging"], check=True)
    subprocess.run(["python3", "-m", "pip", "freeze"], stdout=open("requirements-current.txt", "w"))
    subprocess.run([
        "python3", "bootstrapper.py",
        "--pipeline-name", "data_ingestion_pipeline",
        "--cos-endpoint", minio_endpoint,
        "--cos-bucket", minio_bucket,
        "--cos-directory", cos_directory,
        "--cos-dependencies-archive", "test_responsetime.tar.gz",
        "--file", "llm-rag-deployment/examples/pipelines/test_responsetime.py",
        "--outputs", "responsetime_result.json",
    ], check=True, env={
        **os.environ,
        "ELYRA_RUNTIME_ENV": "kfp",
        "ELYRA_ENABLE_PIPELINE_INFO": "True",
        "ELYRA_WRITABLE_CONTAINER_DIR": "/tmp",
    })


@dsl.component(base_image=RUNTIME_IMAGE)
def summarize(
    minio_endpoint: str = "http://minio.ic-shared-rag-minio.svc:9000",
    minio_bucket: str = "pipeline",
    cos_directory: str = "pipeline-code",
):
    """Summarize pipeline results."""
    import subprocess, os
    os.environ["HOME"] = "/tmp"
    os.makedirs("/tmp/jupyter-work-dir", exist_ok=True)
    os.chdir("/tmp/jupyter-work-dir")

    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/bootstrapper.py",
        "--output", "bootstrapper.py"
    ], check=True)
    subprocess.run([
        "curl", "--fail", "-H", "Cache-Control: no-cache", "-L",
        "file:///opt/app-root/bin/utils/requirements-elyra.txt",
        "--output", "requirements-elyra.txt"
    ], check=True)

    subprocess.run(["python3", "-m", "pip", "install", "packaging"], check=True)
    subprocess.run(["python3", "-m", "pip", "freeze"], stdout=open("requirements-current.txt", "w"))
    subprocess.run([
        "python3", "bootstrapper.py",
        "--pipeline-name", "data_ingestion_pipeline",
        "--cos-endpoint", minio_endpoint,
        "--cos-bucket", minio_bucket,
        "--cos-directory", cos_directory,
        "--cos-dependencies-archive", "summarize_results.tar.gz",
        "--file", "llm-rag-deployment/examples/pipelines/summarize_results.py",
        "--inputs", "responsetime_result.json",
        "--outputs", "results.json",
    ], check=True, env={
        **os.environ,
        "ELYRA_RUNTIME_ENV": "kfp",
        "ELYRA_ENABLE_PIPELINE_INFO": "True",
        "ELYRA_WRITABLE_CONTAINER_DIR": "/tmp",
    })


@dsl.pipeline(name="data-ingestion-response-check", description="Data ingestion and response check pipeline")
def data_ingestion_response_check_pipeline():
    ingest_task = data_ingestion()
    ingest_task.set_env_variable("AWS_ACCESS_KEY_ID", "minio")
    ingest_task.set_env_variable("AWS_SECRET_ACCESS_KEY", "minio123")

    query_task = data_query()
    query_task.set_env_variable("AWS_ACCESS_KEY_ID", "minio")
    query_task.set_env_variable("AWS_SECRET_ACCESS_KEY", "minio123")
    query_task.after(ingest_task)

    resp_task = test_response_time()
    resp_task.set_env_variable("AWS_ACCESS_KEY_ID", "minio")
    resp_task.set_env_variable("AWS_SECRET_ACCESS_KEY", "minio123")
    resp_task.after(query_task)

    sum_task = summarize()
    sum_task.set_env_variable("AWS_ACCESS_KEY_ID", "minio")
    sum_task.set_env_variable("AWS_SECRET_ACCESS_KEY", "minio123")
    sum_task.after(resp_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        data_ingestion_response_check_pipeline,
        "/tmp/data_ingestion_response_check_v2.yaml",
    )
    print("Compiled: /tmp/data_ingestion_response_check_v2.yaml")
