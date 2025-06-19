FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y python3.12 python3-pip

WORKDIR /opt/mlflow

COPY MLProject/requirements.txt .
RUN pip install -r requirements.txt

COPY mlruns /opt/mlflow/mlruns

RUN python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/mlflow/mlruns/0/${{ env.RUN_ID }}/artifacts/model', install_mlflow=False, enable_mlserver=False, env_manager='virtualenv')"

CMD ["mlflow", "models", "serve", "--model-uri", "runs:/${{ env.RUN_ID }}/model", "--host", "0.0.0.0", "--port", "8000"]
