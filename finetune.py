# %%
import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
# from transformers import LlamaForCausalLM as Model
# from transformers import LlamaTokenizer as Tokenizer
# from transformers import GPT2LMHeadModel as Model
# from transformers import GPT2Tokenizer as Tokenizer
# from transformers import GPTJForCausalLM as Model
from transformers import AutoModelForCausalLM as Model
from transformers import AutoTokenizer as Tokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 24 # this could actually be 5 but i like powers of 2
BATCH_SIZE = 256 
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
SAVE_EVAL_STEPS = 342 
EPOCHS = 2  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
# TARGET_MODULES = [
#     "c_attn",
# ]
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
# DATA_PATH = "data/alpaca_data_cleaned.json"
DATA_PATH = "data/unified_chip2.json"
# OUTPUT_DIR = "lora/gpt2-mod"
# BASE_MODEL = "gpt2-xl"
OUTPUT_DIR = "lora/gptj-chip2"
BASE_MODEL = "EleutherAI/gpt-j-6B"
# OUTPUT_DIR = "lora/alpaca"
# BASE_MODEL = "decapoda-research/llama-7b-hf"
CHECKPOINT_DIR = None
CHECKPOINT_DIR = "lora/gptj-chip2/checkpoint-1026"

# device_map = {"":"cuda"}
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# %%
model = Model.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    # revision="float16",
    revision="sharded",
    device_map=device_map,
    torch_dtype=torch.float16,
)
# %%
tokenizer = Tokenizer.from_pretrained(
    BASE_MODEL,
    add_eos_token=True
)
# %%
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

# %%

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# %%
data = load_dataset("json", data_files=DATA_PATH)
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

# %%
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS if VAL_SET_SIZE > 0 else None,
        save_steps=SAVE_EVAL_STEPS,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)

model.save_pretrained(OUTPUT_DIR)

print("\n If there's a warning about missing keys above, please disregard :)")

'''
RUN 1
{'loss': 2.982, 'learning_rate': 5.9999999999999995e-05, 'epoch': 0.05}
{'loss': 2.6149, 'learning_rate': 0.00011999999999999999, 'epoch': 0.1}
{'loss': 1.902, 'learning_rate': 0.00017999999999999998, 'epoch': 0.15}
{'loss': 1.4811, 'learning_rate': 0.00023999999999999998, 'epoch': 0.21}
{'loss': 1.4045, 'learning_rate': 0.0003, 'epoch': 0.26}
{'loss': 1.3708, 'learning_rate': 0.0002793103448275862, 'epoch': 0.31}
{'loss': 1.3614, 'learning_rate': 0.00025862068965517237, 'epoch': 0.36}
{'loss': 1.3349, 'learning_rate': 0.00023793103448275862, 'epoch': 0.41}
{'loss': 1.3262, 'learning_rate': 0.00021724137931034481, 'epoch': 0.46}
{'loss': 1.3256, 'learning_rate': 0.000196551724137931, 'epoch': 0.51}
{'eval_loss': 1.2918754816055298, 'eval_runtime': 254.224, 'eval_samples_per_second': 7.867, 'eval_steps_per_second': 0.983, 'epoch': 0.51}
{'loss': 1.3387, 'learning_rate': 0.0001758620689655172, 'epoch': 0.56}
{'loss': 1.3191, 'learning_rate': 0.00015517241379310346, 'epoch': 0.62}
{'loss': 1.2953, 'learning_rate': 0.00013448275862068965, 'epoch': 0.67}
{'loss': 1.3104, 'learning_rate': 0.00011379310344827585, 'epoch': 0.72}
{'loss': 1.2959, 'learning_rate': 9.310344827586206e-05, 'epoch': 0.77}
{'loss': 1.3058, 'learning_rate': 7.241379310344827e-05, 'epoch': 0.82}
{'loss': 1.3052, 'learning_rate': 5.172413793103448e-05, 'epoch': 0.87}
{'loss': 1.2918, 'learning_rate': 3.1034482758620685e-05, 'epoch': 0.92}
{'loss': 1.2834, 'learning_rate': 1.0344827586206895e-05, 'epoch': 0.97}
{'train_runtime': 23465.5637, 'train_samples_per_second': 2.128, 'train_steps_per_second': 0.017, 'train_loss': 1.5128044593028533, 'epoch': 1.0}

From Checkpoint-200:
{'loss': 1.3172, 'learning_rate': 0.00024705882352941174, 'epoch': 0.56}
{'loss': 1.3098, 'learning_rate': 0.000238235294117647, 'epoch': 0.62}
{'loss': 1.3001, 'learning_rate': 0.0002294117647058823, 'epoch': 0.67}
{'loss': 1.3037, 'learning_rate': 0.00022058823529411765, 'epoch': 0.72}
{'loss': 1.3057, 'learning_rate': 0.00021176470588235295, 'epoch': 0.77}
{'loss': 1.3051, 'learning_rate': 0.00020294117647058822, 'epoch': 0.82}
{'loss': 1.291, 'learning_rate': 0.0001941176470588235, 'epoch': 0.87}
{'loss': 1.3002, 'learning_rate': 0.0001852941176470588, 'epoch': 0.92}
{'loss': 1.3015, 'learning_rate': 0.0001764705882352941, 'epoch': 0.97}
{'eval_loss': 1.2573083639144897, 'eval_runtime': 251.6721, 'eval_samples_per_second': 7.947, 'eval_steps_per_second': 0.993, 'epoch': 1.0}
{'loss': 1.29, 'learning_rate': 0.0001676470588235294, 'epoch': 1.03}
{'loss': 1.2724, 'learning_rate': 0.0001588235294117647, 'epoch': 1.08}
{'loss': 1.2864, 'learning_rate': 0.00015, 'epoch': 1.13}
{'loss': 1.2787, 'learning_rate': 0.00014117647058823528, 'epoch': 1.18}
{'loss': 1.2889, 'learning_rate': 0.00013235294117647058, 'epoch': 1.23}
{'loss': 1.2913, 'learning_rate': 0.00012352941176470587, 'epoch': 1.28}
{'loss': 1.2723, 'learning_rate': 0.00011470588235294115, 'epoch': 1.33}
{'loss': 1.2762, 'learning_rate': 0.00010588235294117647, 'epoch': 1.38}
{'loss': 1.2658, 'learning_rate': 9.705882352941176e-05, 'epoch': 1.44}
{'loss': 1.2627, 'learning_rate': 8.823529411764705e-05, 'epoch': 1.49}
{'eval_loss': 1.2454477548599243, 'eval_runtime': 249.8646, 'eval_samples_per_second': 8.004, 'eval_steps_per_second': 1.001, 'epoch': 1.5}
{'loss': 1.2696, 'learning_rate': 7.941176470588235e-05, 'epoch': 1.54}
{'loss': 1.2732, 'learning_rate': 7.058823529411764e-05, 'epoch': 1.59}
{'loss': 1.2744, 'learning_rate': 6.176470588235294e-05, 'epoch': 1.64}
{'loss': 1.2533, 'learning_rate': 5.294117647058824e-05, 'epoch': 1.69}
{'loss': 1.2513, 'learning_rate': 4.4117647058823526e-05, 'epoch': 1.74}
{'loss': 1.2763, 'learning_rate': 3.529411764705882e-05, 'epoch': 1.79}
{'loss': 1.254, 'learning_rate': 2.647058823529412e-05, 'epoch': 1.85}
{'loss': 1.2603, 'learning_rate': 1.764705882352941e-05, 'epoch': 1.9}
{'loss': 1.2745, 'learning_rate': 8.823529411764705e-06, 'epoch': 1.95}
{'loss': 1.2847, 'learning_rate': 0.0, 'epoch': 2.0}
{'eval_loss': 1.2414301633834839, 'eval_runtime': 251.9393, 'eval_samples_per_second': 7.938, 'eval_steps_per_second': 0.992, 'epoch': 2.0}
{'loss': 1.2669, 'learning_rate': 0.00010373831775700933, 'epoch': 2.05}
{'loss': 1.2524, 'learning_rate': 9.813084112149531e-05, 'epoch': 2.1}
{'loss': 1.2719, 'learning_rate': 9.25233644859813e-05, 'epoch': 2.15}
{'loss': 1.2594, 'learning_rate': 8.691588785046728e-05, 'epoch': 2.21}
{'loss': 1.2676, 'learning_rate': 8.130841121495326e-05, 'epoch': 2.26}
{'loss': 1.2628, 'learning_rate': 7.570093457943924e-05, 'epoch': 2.31}
{'loss': 1.2608, 'learning_rate': 7.009345794392522e-05, 'epoch': 2.36}
{'loss': 1.2622, 'learning_rate': 6.44859813084112e-05, 'epoch': 2.41}
{'loss': 1.2604, 'learning_rate': 5.887850467289719e-05, 'epoch': 2.46}
{'eval_loss': 1.2390713691711426, 'eval_runtime': 258.5533, 'eval_samples_per_second': 7.735, 'eval_steps_per_second': 0.967, 'epoch': 2.5}
{'loss': 1.2619, 'learning_rate': 5.327102803738317e-05, 'epoch': 2.51}
{'loss': 1.2467, 'learning_rate': 4.766355140186915e-05, 'epoch': 2.56}
{'loss': 1.2782, 'learning_rate': 4.2056074766355134e-05, 'epoch': 2.62}
{'loss': 1.264, 'learning_rate': 3.6448598130841115e-05, 'epoch': 2.67}
{'loss': 1.2628, 'learning_rate': 3.0841121495327096e-05, 'epoch': 2.72}
{'loss': 1.2593, 'learning_rate': 2.523364485981308e-05, 'epoch': 2.77}
{'loss': 1.2654, 'learning_rate': 1.9626168224299062e-05, 'epoch': 2.82}
{'loss': 1.2573, 'learning_rate': 1.4018691588785045e-05, 'epoch': 2.87}
{'loss': 1.2592, 'learning_rate': 8.411214953271026e-06, 'epoch': 2.92}
{'loss': 1.2521, 'learning_rate': 2.803738317757009e-06, 'epoch': 2.97}
{'eval_loss': 1.2370578050613403, 'eval_runtime': 276.9701, 'eval_samples_per_second': 7.221, 'eval_steps_per_second': 0.903, 'epoch': 3.0}

Mod
{'loss': 2.9894, 'learning_rate': 5.9999999999999995e-05, 'epoch': 0.05}
{'loss': 2.5956, 'learning_rate': 0.00011999999999999999, 'epoch': 0.1}
{'loss': 1.9039, 'learning_rate': 0.00017999999999999998, 'epoch': 0.15}
{'loss': 1.4802, 'learning_rate': 0.00023999999999999998, 'epoch': 0.21}
{'loss': 1.4062, 'learning_rate': 0.0003, 'epoch': 0.26}
{'loss': 1.3858, 'learning_rate': 0.0002793103448275862, 'epoch': 0.31}
{'loss': 1.3446, 'learning_rate': 0.00025862068965517237, 'epoch': 0.36}
{'loss': 1.3502, 'learning_rate': 0.00023793103448275862, 'epoch': 0.41}
{'loss': 1.3318, 'learning_rate': 0.00021724137931034481, 'epoch': 0.46}
{'eval_loss': 1.2940493822097778, 'eval_runtime': 249.7487, 'eval_samples_per_second': 8.008, 'eval_steps_per_second': 1.001, 'epoch': 0.5}
{'loss': 1.3236, 'learning_rate': 0.000196551724137931, 'epoch': 0.51}
{'loss': 1.3156, 'learning_rate': 0.0001758620689655172, 'epoch': 0.56}
{'loss': 1.3045, 'learning_rate': 0.00015517241379310346, 'epoch': 0.62}
{'loss': 1.3098, 'learning_rate': 0.00013448275862068965, 'epoch': 0.67}
{'loss': 1.3032, 'learning_rate': 0.00011379310344827585, 'epoch': 0.72}
{'loss': 1.3125, 'learning_rate': 9.310344827586206e-05, 'epoch': 0.77}
{'loss': 1.2893, 'learning_rate': 7.241379310344827e-05, 'epoch': 0.82}
{'loss': 1.3027, 'learning_rate': 5.172413793103448e-05, 'epoch': 0.87}
{'loss': 1.2926, 'learning_rate': 3.1034482758620685e-05, 'epoch': 0.92}
{'loss': 1.2999, 'learning_rate': 1.0344827586206895e-05, 'epoch': 0.97}
{'eval_loss': 1.272104024887085, 'eval_runtime': 250.4456, 'eval_samples_per_second': 7.986, 'eval_steps_per_second': 0.998, 'epoch': 1.0}
{"loss": 1.2922, "learning_rate": 0.0001676470588235294, "epoch": 1.03}
{"loss": 1.3029, "learning_rate": 0.0001588235294117647, "epoch": 1.08}
{"loss": 1.2923, "learning_rate": 0.00015, "epoch": 1.13}
{"loss": 1.2994, "learning_rate": 0.00014117647058823528, "epoch": 1.18}
{"loss": 1.2944, "learning_rate": 0.00013235294117647058, "epoch": 1.23}
{"loss": 1.291, "learning_rate": 0.00012352941176470587, "epoch": 1.28}
{"loss": 1.2715, "learning_rate": 0.00011470588235294115, "epoch": 1.33}
{"loss": 1.292, "learning_rate": 0.00010588235294117647, "epoch": 1.38}
{"loss": 1.275, "learning_rate": 9.705882352941176e-05, "epoch": 1.44}
{"loss": 1.2872, "learning_rate": 8.823529411764705e-05, "epoch": 1.49}
{"eval_loss": 1.255008578300476, "eval_runtime": 250.5948, "eval_samples_per_second": 7.981, "eval_steps_per_second": 0.998, "epoch": 1.5}
{"loss": 1.2809, "learning_rate": 7.941176470588235e-05, "epoch": 1.54}
{"loss": 1.2772, "learning_rate": 7.058823529411764e-05, "epoch": 1.59}
{"loss": 1.2727, "learning_rate": 6.176470588235294e-05, "epoch": 1.64}
{"loss": 1.2865, "learning_rate": 5.294117647058824e-05, "epoch": 1.69}
{"loss": 1.2738, "learning_rate": 4.4117647058823526e-05, "epoch": 1.74}
{"loss": 1.2776, "learning_rate": 3.529411764705882e-05, "epoch": 1.79}
{"loss": 1.2588, "learning_rate": 2.647058823529412e-05, "epoch": 1.85}
{"loss": 1.2885, "learning_rate": 1.764705882352941e-05, "epoch": 1.9}
{"loss": 1.2722, "learning_rate": 8.823529411764705e-06, "epoch": 1.95}
{"loss": 1.2718, "learning_rate": 0.0, "epoch": 2.0}
{"eval_loss": 1.2495098114013672, "eval_runtime": 250.5357, "eval_samples_per_second": 7.983, "eval_steps_per_second": 0.998, "epoch": 2.0}
{"loss": 1.2809, "learning_rate": 0.00010373831775700933, "epoch": 2.05}
{"loss": 1.2625, "learning_rate": 9.813084112149531e-05, "epoch": 2.1}
{"loss": 1.2854, "learning_rate": 9.25233644859813e-05, "epoch": 2.15}
{"loss": 1.2535, "learning_rate": 8.691588785046728e-05, "epoch": 2.21}
{"loss": 1.272, "learning_rate": 8.130841121495326e-05, "epoch": 2.26}
{"loss": 1.2611, "learning_rate": 7.570093457943924e-05, "epoch": 2.31}
{"loss": 1.2737, "learning_rate": 7.009345794392522e-05, "epoch": 2.36}
{"loss": 1.2786, "learning_rate": 6.44859813084112e-05, "epoch": 2.41}
{"loss": 1.2614, "learning_rate": 5.887850467289719e-05, "epoch": 2.46}
{"eval_loss": 1.2450251579284668, "eval_runtime": 69.762, "eval_samples_per_second": 28.669, "eeval_steps_per_second": 3.584, "epoch": 2.5}
{"loss": 1.2702, "learning_rate": 5.327102803738317e-05, "epoch": 2.51}
{"loss": 1.2639, "learning_rate": 4.766355140186915e-05, "epoch": 2.56}
{"loss": 1.2585, "learning_rate": 4.2056074766355134e-05, "epoch": 2.62}
{"loss": 1.2712, "learning_rate": 3.6448598130841115e-05, "epoch": 2.67}
{"loss": 1.2654, "learning_rate": 3.0841121495327096e-05, "epoch": 2.72}
{"loss": 1.2615, "learning_rate": 2.523364485981308e-05, "epoch": 2.77}
{"loss": 1.276, "learning_rate": 1.9626168224299062e-05, "epoch": 2.82}
{"loss": 1.2643, "learning_rate": 1.4018691588785045e-05, "epoch": 2.87}
{"loss": 1.2633, "learning_rate": 8.411214953271026e-06, "epoch": 2.92}
{"loss": 1.2558, "learning_rate": 2.803738317757009e-06, "epoch": 2.97}
{"eval_loss": 1.2422480583190918, "eval_runtime": 69.471, "eval_samples_per_second": 28.789, "eval_steps_per_second": 3.599, "epoch": 3.0}
'''
'''gptj
{'loss': 2.209, 'learning_rate': 5.9999999999999995e-05, 'epoch': 0.1}
{'loss': 1.6373, 'learning_rate': 0.00011999999999999999, 'epoch': 0.2}
{'loss': 1.1924, 'learning_rate': 0.00017999999999999998, 'epoch': 0.31}
{'loss': 1.1448, 'learning_rate': 0.00023999999999999998, 'epoch': 0.41}
{'loss': 1.1102, 'learning_rate': 0.0003, 'epoch': 0.51}
{'loss': 1.0937, 'learning_rate': 0.00023684210526315788, 'epoch': 0.61}
{'loss': 1.0849, 'learning_rate': 0.0001736842105263158, 'epoch': 0.72}
{'loss': 1.0905, 'learning_rate': 0.00011052631578947366, 'epoch': 0.82}
{'loss': 1.0789, 'learning_rate': 4.7368421052631574e-05, 'epoch': 0.92}
{'eval_loss': 1.070764183998108, 'eval_runtime': 172.6338, 'eval_samples_per_second': 11.585, 'eval_steps_per_second': 1.448, 'epoch': 1.0}
'''
# %%
