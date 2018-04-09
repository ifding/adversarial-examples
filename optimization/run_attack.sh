#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

#INPUT_DIR=$1
#OUTPUT_DIR=$2
#MAX_EPSILON=$3

# More iterations gives a stronger attack; however, runtime is very limited...
#NUM_ITERATIONS=15

#python attack_iter_target_class.py \
#  --input_dir="${INPUT_DIR}" \
#  --output_dir="${OUTPUT_DIR}" \
#  --max_epsilon="${MAX_EPSILON}" \
#  --num_iter="${NUM_ITERATIONS}"


python optimization.py -i test -o output/GoogleNet --model GoogleNet --file_list test/test_file_list.txt
