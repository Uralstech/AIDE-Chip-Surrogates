export TOTAL_INSTANCES=8
export INSTANCE_INDEX=0 # Set per instance
export S3_BUCKET=
export AWS_REGION=

cd /home/ubuntu
python3 run_simulations.py --config configs_15k.csv --workers 56