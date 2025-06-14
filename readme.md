# llm-power-testing

In this repository is a number of docker compose files that are used to build containers used for stress testing 5090 GPUs using a number of various technologies.

The main programs include:

* [gpu-burn](https://github.com/wilicc/gpu-burn)
* [tensorRT](https://github.com/NVIDIA/TensorRT-LLM)
* [inference-benchmarker](https://github.com/huggingface/inference-benchmarker)
* TO BE ADDED

## GPU Burn

For a simple GPU stress test, you can run the `gpu-burn` container which should stress out every gpu in the system.

```bash
cd gpu-burn
docker compose up
```

## Tensor RT

According to [this](https://github.com/NVIDIA/TensorRT-LLM/issues/2953) git issue, 5090's have issues running with tensor or pipeline parallelism enabled in tensorRT. Because of that, the tensorRT container in this repository only stresses a single GPU. Additionally, the tensorRT program just hosts a webserver with the openAI API. This means that once tensorRT is running, we need to additionally run the inference-benchmarker program which generates a stress for the device.

> [!note]
> The TensorRT setup mostly comes from [this](https://github.com/NVIDIA/TensorRT-LLM/issues/4275#issuecomment-2941155249) comment on github

To run TensorRT, it first needs a llm model, of which the default is Deepseek-R1:1.5B from Hugging Face. You *should* be able to run other models by cloning them into the repository and changing the mounts in the docker-compose.

```bash
cd tensorRT
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
docker compose up
```

To run inference-benchmarker:
```bash
cd inference-benchmarker
docker compose up
```

## Building Containers

Many of the containers in this repository need to be manually built. You can do so by following these instructions.

For [gpu-burn](https://github.com/wilicc/gpu-burn) and [inference-benchmarker](https://github.com/huggingface/inference-benchmarker), the containers can be built by following the instructions on their respective repos.

For tensorRT, the compose file should justworkTM.

For VLLM, please see more detailed building instructions (fixme if I end up using it).
