unset KUBERNETES_PORT

pip install -e /home/nannau/light_container/ClimatExML

unset KUBERNETES_PORT

mkdir /home/nannau/data

python /home/nannau/light_container/ClimatExML/ClimatExML/train.py