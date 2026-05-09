#!/bin/bash -l

python -m spacy download en

MODEL_NAME=qwen3-32B
EVAL_MODEL_NAME=qwen3-32B
SEED=2025
N=5
REFINE_MAX=1
RUN_NAME=N$N.seed$SEED

echo "Model: $MODEL_NAME"
echo "Run: $RUN_NAME"

VLLM_CONFIGURE_LOGGING=0 python passage_generation.py \
                                --drafter_n $N \
                                --seed $SEED \
                                --revision_max_round 20 \
                                --refinement_max_round $REFINE_MAX \
                                --model_nickname $MODEL_NAME \
                                --run_name $RUN_NAME \
                                --data_path data/Brown.test.json \
                                --constraint_path data/difficulty_series.verified.json

VLLM_CONFIGURE_LOGGING=0 python option_generation.py \
                                --seed $SEED \
                                --drafter_n $N \
                                --revision_max_round 100 \
                                --model_nickname $MODEL_NAME \
                                --drafter_use_exemplars \
                                --reviser_use_exemplars \
                                --neutrality_evaluation_model $EVAL_MODEL_NAME \
                                --factuality_evaluation_model $EVAL_MODEL_NAME \
                                --complexity_evaluation_model $EVAL_MODEL_NAME \
                                --passage_file output/passage_generation/$MODEL_NAME-$RUN_NAME/final-MaxStep$REFINE_MAX/all_results.json \
                                --run_name $RUN_NAME
