<div align="center">
  <p align="center">
    <a href=""><img src="figures/main_figure.svg" alt="Main figure" width="90%"></a>
  </p>
  <h2>KV Cache Steering for Controlling Frozen LLMs</h2>
  <b>Authors:</b> Max Belitsky, Dawid Kopiczko, Michael Dorkenwald, Jehanzeb Mirza, Cees Snoek, Yuki Asano
  
</div>

---

## Overview

**Cache Steering** is a lightweight method for steering the behavior of language models by modifying their key-value cache with a single intervention. In our paper, we use this approach to induce **chain-of-thought reasoning** in small models by injecting steering vectors derived from GPT-4o-generated reasoning traces.

Key advantages of cache steering:
- **Seamless integration with standard APIs**: Works with existing transformer models and inference pipelines.
- **Behavioral control**: Allows inducing reasoning behaviors and stylistic shifts.
- **No model or code changes required**: Applies post-hoc intervention without retraining or architectural modifications (no hooks!).
- **Minimal latency overhead**: Fast one-shot intervention with negligible runtime cost.

Read our paper [here](http://arxiv.org/abs/2507.08799) for full details.

## Cache Steering in Action

### Example 1: reasoning induction
These are the examples from the [notebook](examples.ipynb):
- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Prompt: `What is the capital of France?`

| **No intervention** | **Cache Steering** |
| --- | --- |
| *"The capital of France is Paris."* | *"When it comes to the capital of France, it is essential to note that the official capital of France is Paris, which is also the largest city in the country. Paris is not only the capital but also the economic, cultural, and political center of France. In terms of geography, Paris is located in the northern part of France, in the region of Nord-Pas-de-Calais. It is situated on the Seine River, which flows through the city, and is surrounded by several other major cities, including Lyon, Marseille, and Reims. In terms of population, Paris has..."* |

### Example 2: style transfer

| **No intervention** | **Cache Steering (analogical reasoning)** |
| --- | --- |
| *"The capital of France is Paris."* | *"Just like how you have a home, a country has a capital. The capital of France is Paris. Paris is a big city in France, and it's where the Eiffel Tower is located. It's a beautiful city with lots of history and culture."* |

See [results](results/) for full generations on ARC-c, GSM8K, CSQA and PIQA datasets for all experiments.

## Quick Start

We provide a [notebook](examples.ipynb) with examples on how to use our codebase and a **pure PyTorch + Transformers implementation** from scratch to demonstrate how easy it is to integrate cache steering into any workflow.

### Minimal usage example

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.data.contrastive_dataset import ContrastiveDatasetConstructor
from src.data.loader import load_task_dataset
from src.steering.cache_steering import extract_steering_kv, generate_with_cache_steering
from src.steering.config import SteeringConfig
from src.utils.constants import Tasks

# Load your model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"          # Small model that is fast to run on CPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the config for dataset creation
config = SteeringConfig(
    n_contrastive_samples=10,
    num_fewshot_examples=5,
    tokenizer=tokenizer,
    add_generation_prompt=True,
)

# Load and preprocess the data
task = Tasks.arc_oai
dataset = load_task_dataset(task)

constructor = ContrastiveDatasetConstructor(
    dataset["train"],
    config,
    task=task,
)
contrastive_dataset = constructor.construct_dataset()

# You can define your own dataset as follows (don't forget to apply the chat template to yuor examples):
# positive_examples = ["Chat-templated positive example 1", ...]
# negative_examples = ["Chat-templated negative example 1", ...]
# contrastive_dataset = Dataset.from_dict({
#     "positive": positive_examples,
#     "negative": negative_examples,
# })

# Exctract the steering vectors for each layer
steering_kv = extract_steering_kv(
    model=model,
    tokenizer=tokenizer,
    data=contrastive_dataset,
    steering_config=config,
)

# Generate with cache steering
messages = [{"role": "user", "content": "What is the capital of France?"}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
tokens = tokenizer(input_text, return_tensors='pt').to(device)

steering_config = SteeringConfig(
    tokenizer=tokenizer,
    c_keys=0.0,                     # The steering coefficient for the keys        
    c_values=10,                    # The steering coefficient for the values
    append_special_token=True,      # Whether to append a special token to the input to offset the position of the steering token
)
generation_kwargs = {"max_new_tokens": 100, "do_sample": False}

output = generate_with_cache_steering(
    model,
    tokens["input_ids"],
    steering_kv=steering_kv,
    steering_config=steering_config,
    attention_mask=tokens["attention_mask"],
    **generation_kwargs,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```



## Project Structure

- `local_data/` – data with CoT reasoning traces for different tasks
- `cached_vectors/` – precomputed steering vectors will be saved here
- `results/` – experiment outputs, metrics, and generations
- `src/` – core cache steering implementation
- `jobs/` – scripts and SLURM templates for experiments



## Replicating Experiments
For full instructions to replicate our experiments, see [INSTRUCTIONS.md](INSTRUCTIONS.md). Below is a small example:

**Generate experiment files** (SLURM example)
The below example script generates a separate file for each experiment for cache steering evaluation:
```bash
python jobs/scripts/generate_jobs.py --config jobs/configs/best_args.yaml \
  --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/steering_results \
  --experiment_name steering" --time 01:00:00
```



## Citation
You can cite our work like this:
```bibtex
@article{belitsky2025kv,
  title={Kv Cache Steering for Controlling Frozen LLMs},
  author={Belitsky, Max and Kopiczko, Dawid J and Dorkenwald, Michael and Mirza, M Jehanzeb and Glass, James R and Snoek, Cees GM and Asano, Yuki M},
  journal={arXiv preprint arXiv:2507.08799},
  year={2025}
}
```
