#!/usr/bin/env bash

# Compile for tensorRT's specific quantization format
python3 /app/examples/quantization/quantize.py \
  --model_dir /app/examples/deepseek/ \
  --dtype bfloat16 \
  --qformat fp8 \
  --kv_cache_dtype fp8 \
  --output_dir /ckpt \
  --calib_size 512

# Build the LLM engine
trtllm-build --checkpoint_dir /ckpt --output_dir /engine \
  --remove_input_padding enable \
  --kv_cache_type paged \
  --max_batch_size 2048 \
  --max_input_len 1024 \
  --max_num_tokens 1024 \
  --max_seq_len 2048 \
  --use_paged_context_fmha enable \
  --use_fp8_context_fmha enable \
  --gemm_plugin disable \
  --multiple_profiles enable

mkdir /triton_model_repo
cp -r /app/all_models/inflight_batcher_llm/* /triton_model_repo/

# Not 100% sure on these, haven't bothered looking it up. JustworksTM
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE} && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT} && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:True && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT} && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}

# Serve the Model
trtllm-serve ${ENGINE_DIR} --tokenizer ${TOKENIZER_DIR}
