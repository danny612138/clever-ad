WISEAD_MODEL=$1
OUTPUT_PATH=$2
# generate predictions
python eval/carla/run_inference.py --model_path ${WISEAD_MODEL} --output_path ${OUTPUT_PATH} --image_folder ./data/carla/DATASET