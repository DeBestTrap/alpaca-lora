# %%
from typing import List, Any, Iterable, TypeVar, overload, Union
import os
import torch
import gc
import sys
import json
import gradio as gr
import transformers
import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import collections.abc
from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# TODO figure out how to automatically determine the device
device = "cuda"
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

_T = TypeVar("_T")
_SmarterSet = TypeVar("_SmarterSet")

class SmarterSet(set):
    """
    Same as a regular python set but the set will return itself when using add() or clear().
    """
    # TODO Move to its own file
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, __iterable: Iterable[_T]) -> None:
        ...

    def __str__(self) -> str:
        ...

    def add(self, __element: Any) -> _SmarterSet:
        ...

    def clear(self) -> _SmarterSet:
        ...

    def __init__(self, __arg1: Union[Iterable[_T], None] = None) -> None:
        if __arg1 == None:
            super().__init__()
        elif isinstance(__arg1, collections.abc.Iterable):
            super().__init__(__arg1)
        else:
            raise TypeError("Cannot create a SmarterSet from a non-iterable.")

    def __str__(self) -> str:
        # TODO theres probably a better and faster way to do this
        return "{" + ", ".join([f"{e}" for e in self]) + "}"

    def add(self, __element: Any) -> _SmarterSet:
        '''
        Add an element to a set and return the set.

        This has no effect if the element is already present.
        '''
        super().add(__element)
        return self

    def clear(self) -> _SmarterSet:
        '''
        Clear the set and return the set.
        '''
        super().clear()
        return self


def create_model(version: str,
                 revision: str,
                 load_8bit: str,
                 progress=gr.Progress(track_tqdm=True)):
    '''
    Loads a pretrianed model stored in a global variable named model.

    If a model already is loaded then the old model will be replaced by the new one.

    Return:
        A str containing the version and revision of the model
    '''
    global model
    global tokenizer
    global loaded_model_text

    if not version or not revision:
        return "Invalid Model"

    try:
        if model:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except NameError:
        pass

    if load_8bit == "True":
        load_8bit_bool = True
    else:
        load_8bit_bool = False

    model = AutoModelForCausalLM.from_pretrained(
        version,
        revision=revision,
        load_in_8bit=load_8bit_bool,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(version, add_eos_token=True)
    loaded_model_text = f"{version}, {revision}"
    return loaded_model_text


def prepare_model_for_generating(load_8bit: str,
                                 lora_weights: str,
                                 progress=gr.Progress(track_tqdm=True)):
    '''
    Load a PEFT model with a LoRA for generating text

    Returns:
        A str containing the version and revision of the model
    '''
    global model
    global loaded_model_text

    if lora_weights == "":
        return f"Generating: {loaded_model_text}"

    if load_8bit == "True":
        load_8bit_bool = True
    else:
        load_8bit_bool = False

    model = PeftModel.from_pretrained(model,
                                      os.path.join("lora", lora_weights),
                                      torch_dtype=torch.float16,
                                      adapter_name="1",
                                      device_map="auto")

    # TODO ADD MULTI LORA LOADING
    # model.load_adapter("lora\EleutherAI-gpt-j-6B-sharded-unified_chip2", adapter_name="2")
    if not load_8bit_bool:
        model.half()  # seems to fix bugs for some users.
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return f"Generating: {loaded_model_text}, {lora_weights}"


def prepare_model_for_training(target_modules: List[str],
                               progress=gr.Progress(track_tqdm=True)):
    '''
    Load a PEFT model with hyperparameters to train a LoRA

    Returns:
        A str containing the version and revision of the model
    '''
    global model
    global loaded_model_text
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    return f"Finetuning: {loaded_model_text}"


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
    '''
    This function masks out the labels for the input,
    so that our loss is computed only on the response.

    Returns:
        Dictionary containing input ids, labels and attention mask
    '''
    user_prompt = ((
        f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
    ) if data_point["input"] else (
        f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""))
    len_user_prompt_tokens = (len(
        tokenizer(
            user_prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
        )["input_ids"]) - 1)  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids":
        full_tokens,
        "labels":
        [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


def train(data_path: str,
          lora_name: str,
          checkpoint: str,
          batch_size: int,
          micro_batch_size: int,
          save_eval_steps: int,
          epochs: int,
          learning_rate: int,
          val_set_size: int,
          progress=gr.Progress(track_tqdm=True)):
    '''
    Trains a LoRA and saves it in the lora directory

    Returns:
        A str indicating the result of the training
    '''
    global model
    try:
        assert(model)
    except:
        return "Wait for model to load before using"
    global loaded_model_text

    if lora_name == "New Lora":
        lora_name = "-".join(loaded_model_text.split(", "))

    lora_name = lora_name.replace("/", "-")
    lora_name = lora_name.replace("\\", "-")
    output_dir = os.path.join("lora", lora_name)

    if checkpoint == "":
        checkpoint_dir = None
    else:
        checkpoint_dir = os.path.join("lora", lora_name, checkpoint)

    print(lora_name)
    print(checkpoint_dir)

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    data = load_dataset("json", data_files=os.path.join("data", data_path))
    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size,
                                                   shuffle=True,
                                                   seed=42)
        train_data = train_val["train"].shuffle().map(
            generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(
            generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=save_eval_steps if val_set_size > 0 else None,
            save_steps=save_eval_steps,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                                   mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict())).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=checkpoint_dir)

    # TODO check
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    return f"LoRA saved at {output_dir}"


def load_models_json():
    '''
    Get parameters for loading models

    Returns:
        A dictionary containing parameters related to loading models.
    '''
    with open("choices_models.json") as f:
        return json.load(f)


def generate_prompt(instruction, input=None):
    '''
    Returns:
        A prompt formatted with instruction (and input if provided)
    '''
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def generate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.2,
        length_penalty=1,
        ngram_size=0,
        max_new_tokens=128,
        use_generation_config=True,
        progress=gr.Progress(track_tqdm=True),
        **kwargs,
):
    '''
    Generate a response to the instruction provided.

    Reurns:
      A string a with a response to the instruction provided
    '''
    global model
    try:
        assert(model)
    except:
        return "Wait for model to load before using"
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    if use_generation_config:
        generation_config = GenerationConfig(
            pad_token_id=-1,
            bos_token_id=0,
            eos_token_id=2,
            # eos_token_id=2,
            # do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=ngram_size,
            # max_new_tokens=128,
            # length_penalty=-1.0,
            # early_stopping=True,
            **kwargs,
        )
    else:
        generation_config = None

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # return output
    # print(f"OUTPUT: {output}")
    # print(f"SPLIT: {output.split(instruction)})")
    # output = output.split(instruction)[1].strip()
    # print(f"FIRST: {output}")
    # output = output.split("### Response:")[1].strip()
    # print(f"SECOND: {output}")
    # if "### Instruction:" in output:
    #     return output.split("### Instruction:")[0].strip()
    # return output

    # most_recent_instruction = instruction.split("### Instruction:")[-1]
    # response_splits = output.split("### Instruction:")
    # for str in response_splits:
    #     if most_recent_instruction in str:
    #         return str
    # print(f"----- SCREE\n{output}")
    # print(f"----- INSTR\n{instruction}")
    # return output

    response = output[len(prompt):].split("### Instruction:")[0]
    print(response)
    return response 

    # return output.split("### Response:")[1].strip()
    # return output.split(
    #     "### Response:")[1].strip() + "\n\nDEBUG=========" + output


def evaluate_model(data_path: list,
                   load_8bit: str,
                   lora_weights: str,
                   base_model: str,
                   progress=gr.Progress(track_tqdm=True)):
    '''
    Don't use yet
    TODO Finish this
    '''
    global model
    try:
        assert(model)
    except:
        # Havent tested returning strs to json block
        return "Wait for model to load before using"

    if load_8bit == "True":
        load_8bit_bool = True
    else:
        load_8bit_bool = False

    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    seqeval = evaluate.load("seqeval")
    rouge = evaluate.load("rouge")

    # Loading data and generating the output

    list_of_datasets = []
    for d in data_path:
        data = load_dataset("json", data_files=os.path.join("data", d))
        list_of_datasets.append(data["train"])
    data = concatenate_datasets(list_of_datasets)

    references = list(map(lambda x: [x], data["output"]))
    results_file_output = dict()
    results_file_output["info"] = {
        "LOAD_8BIT": load_8bit_bool,
        "DATA_PATH": data_path,
        "LORA_WEIGHTS": lora_weights,
        "BASE_MODEL": base_model
    }
    predictions = list(
        map(lambda x, y: generate(x, y, use_generation_config=False),
            progress.tqdm(data["instruction"], desc="Generating Responses"),
            data["input"]))

    # BLEU
    results = bleu.compute(predictions=predictions, references=references)
    results_file_output["bleu"] = results

    # BERT
    results = bertscore.compute(predictions=predictions,
                                references=references,
                                lang="en")
    results_file_output["bertscore"] = results

    # str = "| "
    # for label in results:
    #     str += f"{label:<10}| "
    # print(str.strip())
    # str = "| "
    # for key, val in results.items():
    #     str += f"{val[0]:<10.07}| "
    # print(str.strip())

    # # seqeval
    # results = seqeval.compute(predictions=predictions, references=references)
    # results_file_output["seqeval"]= results

    # ROUGE
    results = rouge.compute(predictions=predictions, references=references)
    results_file_output["rouge"] = results

    import json
    with open("results.json", "w") as f:
        json.dump(results_file_output, f)

    results_predictions = dict()
    results_predictions["info"] = {
        "LOAD_8BIT": load_8bit_bool,
        "DATA_PATH": data_path,
        "LORA_WEIGHTS": lora_weights,
        "BASE_MODEL": base_model
    }
    results_predictions["data"] = data
    results_predictions["data"]["predictions"] = predictions

    with open("results_predictions.json", "w") as f:
        json.dump(results_predictions, f)

    print(results_file_output)
    print(type(results_file_output))
    return results_file_output


def generate_prompt_boolq(passage:str, instruction:str, prompt_style:str):
    '''
    Generates a prompt depending on the prompt_style provided.

    Sources:
        https://en.wikipedia.org/wiki/Final_Fantasy_XIV

    Args:
        prompt_style can be "Zero-Shot", "One-Shot", "Few-Shot"
    Returns:
        A str containing the prompt
    '''

    if prompt_style == "Zero-Shot":
        return f'''Based on this passage:
{passage}

{instruction}'''

    elif prompt_style == "One-Shot":
        return f'''Based on this passage:
Final Fantasy XIV is a massively multiplayer online role-playing game (MMORPG) developed and published by Square Enix. Directed and produced by Naoki Yoshida, it was released worldwide for PlayStation 3 and Windows in August 2013, as a replacement for the failed 2010 version of the game, with support for PlayStation 4, OS X, and PlayStation 5 releasing later.

is Final Fantasy XIV made by Square Enix

### Response:
Yes, Final Fantasy XIV is developed by Square Enix.

### Instruction:
Based on this passage:
{passage}

{instruction}'''

    elif prompt_style == "Few-Shot":
        return f'''Based on this passage:
Final Fantasy XIV is a massively multiplayer online role-playing game (MMORPG) developed and published by Square Enix. Directed and produced by Naoki Yoshida, it was released worldwide for PlayStation 3 and Windows in August 2013, as a replacement for the failed 2010 version of the game, with support for PlayStation 4, OS X, and PlayStation 5 releasing later.

is Final Fantasy XIV made by Square Enix

### Response:
Yes, Final Fantasy XIV is developed by Square Enix.

### Instruction:
Based on this passage:
{passage}

{instruction}'''

    else:
        raise Exception()


def generate_boolq_responses(prompt_style:str, progress=gr.Progress(track_tqdm=True)):
    '''
    Generates responses for the BoolQ dataset.

    Args:
        prompt_style can be "Zero-Shot", "One-Shot", "Few-Shot"
    Returns:
        A str indicating the result of the generation
    '''
    global model
    try:
        assert(model)
    except:
        return "Wait for model to load before using"

    data = load_dataset("boolq")
    data = data["validation"]

    instructions = progress.tqdm( list(map(lambda p, q:generate_prompt_boolq(p, q, prompt_style), data["passage"], data["question"])), desc="Generating Responses")
    
    predictions = list(
        map(
            lambda x: generate(x, input=None, use_generation_config=False, max_new_tokens=5),
            instructions))

    results_predictions = dict()
    results_predictions["predictions"] = predictions
    with open("results_predictions_boolq.json", "w") as f:
        json.dump(results_predictions, f)
    return "Responses Finished"

# %%
def evaluate_boolq_responses():
    '''
    Evaluates the responses generated for the BoolQ dataset.

    Returns:
        A str of the accuracy of the responses
    '''
    # TODO add files wtih dropdown
    data = load_dataset("boolq")
    data = data["validation"]
    with open("results_predictions_boolq.json", "r") as f:
        d = json.load(f)
        predictions = d["predictions"]

    correct = 0
    ambi_ctr = 0
    for i in range(len(data)):
        response:str = predictions[i].strip()
        if "yes" in response[0:4].lower():
            if data["answer"][i] == True:
                correct += 1
        elif "no" in response[0:4].lower():
            if data["answer"][i] == False:
                correct += 1
        else:
            ambi_ctr += 1
            print(f'ambi! {response}')

    err = correct / len(data)
    print(f"{err*100:.2f}% - Accuracy")
    print(f"{(correct)*100 / (len(data)-ambi_ctr):.2f}% - Accuracy with ambiguous responses removed")
    return f"{err}"

# boolqe()
# %%
