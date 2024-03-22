nvidia-smi
echo $COMET_API_KEY
pip install -e /project/ClimatExML
source /project/ClimatExML/env_vars.sh
python /project/ClimatExML/ClimatExML/train.py
