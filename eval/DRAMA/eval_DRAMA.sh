WISEAD_MODEL=$1
OUTPUT_PATH=$2
# generate predictions
python eval/DRAMA/run_inference.py --model_path ${WISEAD_MODEL} --output_path ${OUTPUT_PATH}
# generate .csv file for evaluation
python eval/DRAMA/generate_csv.py --input_folder_path ${OUTPUT_PATH}/ --output_folder_path ${OUTPUT_PATH}
# run evaluation with DRAMA-Judge metric
python eval/DRAMA/eval_metric.py --predictions_path ${OUTPUT_PATH}/drama_results.csv