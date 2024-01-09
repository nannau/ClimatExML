nvidia-smi
echo $MLFLOW_TRACKING_URI
echo $MLFLOW_EXPERIMENT_NAME
echo $MLFLOW_S3_ENDPOINT_URL
pip install -e /project/ClimatExML
source /project/ClimatExML/mlflow.env
python /project/ClimatExML/ClimatExML/train.py
