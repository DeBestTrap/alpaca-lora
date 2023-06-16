# Alpaca-LoRA WebUI

This is a fork of tloen's Alpaca LoRA with with a webui!

## Getting started with training LoRAs (Low Rank Adaptaion) \[[1](https://arxiv.org/pdf/2106.09685.pdf)\] for LLMs (Large Language Models)

## Licensing Note

Meta's LLaMA model is noncommercial and only for researchers that have applied for the license to use.

Alpaca's Dataset is under `CC By NC 4.0` which means noncommerical too.

## What is LoRA?
It is a type of fine tuning that allows the user to freeze the weights of the base model and train a separate model on top of the base model. This separate model (LoRA) is a much smaller file (in the order of MB) and the original model is unaltered.

If you want more technical information, see the original paper \[[1](https://arxiv.org/pdf/2106.09685.pdf)\].

## Computer Specifications
- An Nvidia GPU is needed to utilize CUDA for bitsandbytes.
  -  Preferably with a large pool of VRAM
- Linux or Windows
- Large pool of RAM may be needed also

## My Specs
These are the specs of the computer I used to finetune the GPT-2 model.
```
Windows 10
Nvidia GTX 1080-Ti (11GB VRAM)
Intel i7-6700k
16GB DDR4-3200
```

I later upgraded my GPU to an `RTX 3090` which has 24GB VRAM and allowed me to load the GPT-J model and finetune it.

## Setting Up Environment
If you do not have CUDA toolkit installed see https://developer.nvidia.com/cuda-downloads.

### Virtual Environment and Installing Packages
Create a virtual environment for this project and enter it:
```bash
python -m venv venv
./venv/Scripts/activate
```
Install the requirements:
```bash
python -m pip install -r requirements.txt
```

You also have to install the correct version of pytorch depending on OS and CUDA toolkit version.
Get the command to install for your system at
https://pytorch.org/get-started/locally/
* Note: At the time of writing the latest CUDA toolkit version is 12.1 and pytorch supports up to 11.7. It is said that NVIDIA makes the toolkit backwards compatible so it should work.

### If you are not using Windows, skip this step
bitsandbytes was compiled for Linux and don't support Windows however someone made a fix.

1. Goto https://github.com/DeXtmL/bitsandbytes-win-prebuilt and download `libbitsandbytes_cuda116.dll`
1. Place the dll into `root\venv\Lib\site-packages\bitsandbytes`
1. Edit `root\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py`
  1. search for: `if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.so', None, None, None, None`
  1. replace with: `if torch.cuda.is_available(): return 'libbitsandbytes_cuda116.dll', None, None, None, None`
  1. search for this twice: `self.lib = ct.cdll.LoadLibrary(binary_path)`
  1. replace with: `self.lib = ct.cdll.LoadLibrary(str(binary_path))`

Source: https://github.com/oobabooga/text-generation-webui/issues/147#issuecomment-1456040134
* Note: this solution was for a webui that we are not using however, the fix is the same.

## How to use the WebUI 

### Generation
1. First, load your model
   1. Select your:
        * Model
        * Version
        * Revision
        * Whether you want to load in as 8 bit
        * (Optional) LoRA model
    1. Hit "Load Model for Generating"
1. Type in your instruction and input
1. Adjust your generation parameters
    * (More information about these parameters below)
1. Hit "Generate"!

### Evaluation TODO NOT FINISHED YET
1. First, load your model
   1. Select your:
        * Model
        * Version
        * Revision
        * Whether you want to load in as 8 bit
        * (Optional) LoRA model
    1. Hit "Load Model for Generating"
1. TODO
1. Hit "Evaluate"!

### Finetuing
1. Click on the finetune tab (the load model tab should change to reflect the change)
1. Load your model
   1. Select your:
        * Model
        * Version
        * Revision
        * Whether you want to load in as 8 bit
        * (Optional) LoRA model
        * (Optional) If you want to continue training a LoRA, select the checkpoint to start from
    1. Hit "Load Model for Finetuning"
1. Adjust your hyperparameters
    * (More information about these hyperparameters below)
1. Select the data to be trained on
1. Hit "Finetune"!

# More information about finetuning, generation, and evaluation with code
## Finetuning a LoRA

Depending on how much VRAM you have will determine what size model you can use and how fast it can train.
* Rational for choosing GPT-2: This model only has 1B parameters and can easily be trained on a consumer GPU (1080-Ti in my case).

Check to see that the correct model class and tokenizer are imported.
```python
from transformers import GPT2LMHeadModel as Model
from transformers import GPT2Tokenizer as Tokenizer
# You could try and use the following imports instead,
# It should work but I'm not sure if changes the speed at which it trains 
# from transformers import AutoModelForCausalLM as Model
# from transformers import AutoTokenizer as Tokenizer
```

### Parameters
These parameters control the size of batches of data to use at a time when training.
```python
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
```

This parameter controls how many steps until a checkpoint of the model is saved and an eval is run on the testing data.
```python
SAVE_EVAL_STEPS = 200
```

How many epochs should it train for
(when training from a checkpoint, the epoch is incremental. For example you trained for 1 epoch and stopped, now you load that checkpoint at a later date and want to train for one more epoch so you set EPOCH = 2)
```python
EPOCHS = 3
```

Hyperparameters for training the LoRA.
```python
# These are set from the original repo (alpaca-lora). I wouldn't change them unless you know exactly what you are doing.
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAR_SET_SIZE = 2000
```

This parameter needs to change based on based on which model you are using, see https://github.com/huggingface/peft/blob/main/src/peft/mapping.py for all the supported ones.
```python
TARGET_MODULES = [
  "c_attn",
]
```

These parameters are self explanatory.
```python
DATA_PATH = "alpaca_data_cleaned.json"
OUTPUT_DIR = "lora/gpt2"
BASE_MODEL = "gpt2-xl"
CHECKPOINT_DIR = None
# CHECKPOINT_DIR would be in the form of "lora/<model>/checkpoint-<step>" if loading from a checkpoint
```

## Sidenote about using GPT-J-6B
I upgraded to using a 3090 and wanted to try loading a larger model with all the extra VRAM (24GB-GDDR6X).
I was unable to load the full (F32) model and covert it to F16 weights.
I had to use the sharded half (F16) model using the parameter `revision="sharded"` when loading the pretrained model.
```python
model = Model.from_pretrained(
  BASE_MODEL,
  load_in_8bit=True,
  revision="sharded",  # THIS LINE
  device_map=device_map,
  torch_dtype=torch.float16,
)
```
For some reason the model first loads into my RAM and then is moved into the GPU's VRAM. This seems to be the case for GPT-2 and GPT-J but not for LLaMA.

## Generating
Generating text has parameters that control how the output is generated:
```python
temperature=0.1,
top_p=0.75,
top_k=40,
num_beams=4,
repetition_penalty=1.2,
length_penalty=1,
ngram_size=0,
max_new_tokens=128,
```
The first 4 are explained very well in this blog post from huggingface:
https://huggingface.co/blog/how-to-generate

The repetition and length penalty were added to try and reduce long repeated text but I will probably remove it later.

### Side notes:
* GPT-2 seems to generate tons of repeating text and probably is not worth it to use for any llm applications.

## Evaluate TODO I HAVE NOT STARTED/FINISHED THIS YET
https://huggingface.co/docs/evaluate/index
```bash
python -m pip install evaluate scikit-learn
nltk rouge_score
```

# References
[1] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv.org, 16-Oct-2021. [Online]. Available: https://arxiv.org/abs/2106.09685. [Accessed: 28-Mar-2023]. 

# Original repo's README (unedited)
## 🦙🌲🤏 Alpaca-LoRA: Low-Rank LLaMA Instruct-Tuning

- 🤗 **Try the pretrained model out [here](https://huggingface.co/spaces/tloen/alpaca-lora), courtesy of a GPU grant from Huggingface!**
- Share custom LoRA adapters, including adapters for the larger models, [here](https://github.com/tloen/alpaca-lora/issues/52)
- Users have created a Discord server for discussion and support [here](https://discord.gg/prbq284xX5)
- `alpaca-lora-30b` can be used like ChatGPT; see [here](https://twitter.com/algo_diver/status/1637851640027041798)

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
We provide an Instruct model of similar quality to `text-davinci-003` that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research),
and the code is easily extended to the `13b`, `30b`, and `65b` models.

In addition to the training code, which runs within five hours on a single RTX 4090,
we publish a script for downloading and inference on the foundation model and LoRA,
as well as the resulting [LoRA weights themselves](https://huggingface.co/tloen/alpaca-lora-7b/tree/main).
To fine-tune cheaply and efficiently, we use Hugging Face's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

### Setup

1. Install dependencies

```
pip install -r requirements.txt
```

2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
Near the top of this file is a set of hardcoded hyperparameters that you should feel free to modify.
PRs adapting this code to support larger models are always welcome.

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

### Dataset

In addition to `alpaca_data.json`, which contains the original Stanford Alpaca dataset,
we also include `alpaca_data_cleaned.json`, which has been [stripped of various tokenization artifacts](https://github.com/tloen/alpaca-lora/pull/32)
with the help of @gururise.
This file is now used by default in the training script.

@AndriyMulyar has also provided interactive, embedding-based visualizations of the original dataset's [instructions](https://atlas.nomic.ai/map/alpaca_instructions)
and [outputs](https://atlas.nomic.ai/map/alpaca_outputs),
as well as [clusters of bad examples](https://atlas.nomic.ai/map/d2139cc3-bc1c-441c-8d6f-3e6ffbbc2eda/838019ff-8fe2-42ba-809a-d86d2b98cd50/-18.11668742841587/-11.348087116836096/-20.88850316347706/-17.680468640801223/774455612).

### Notes

- We can likely improve our model performance significantly if we had a better dataset. Consider supporting the [LAION Open Assistant](https://open-assistant.io/) effort to produce a high-quality dataset for supervised fine-tuning (or bugging them to release their data).
- We're continually fixing bugs and conducting training runs, and the weights on the Hugging Face Hub are being updated accordingly. In particular, those facing issues with response lengths should make sure that they have the latest version of the weights and code.

### Example outputs

**Instruction**: Tell me about alpacas.

**Alpaca-LoRA**: Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are known for their soft, luxurious fleece, which is used to make clothing, blankets, and other items. Alpacas are herbivores and graze on grasses and other plants. They are social animals and live in herds of up to 20 individuals.

**Stanford Alpaca**: Alpacas are small, fluffy animals related to camels and llamas. They are native to Peru and Bolivia, and were first domesticated around 5,000 years ago. They are kept mainly for their fine, soft fleece, which is used to make knitwear and other garments. Alpacas are herd animals and live in small family groups, led by an older male. They are highly social animals and can form strong bonds with their owners.

**text-davinci-003**: Alpacas are a domesticated species of South American camelid, similar to llamas. They are native to the Andes Mountains in South America and are kept as livestock for their soft, luxurious wool. Alpacas are social animals and live in herds of up to 20 individuals. They are typically kept in small herds of two to five animals, and are relatively easy to care for. Alpacas are herbivores and feed on grass, hay, and other vegetation. They are also known for their gentle and friendly nature, making them popular as pets.

---

**Instruction**: Tell me about the president of Mexico in 2019.

**Alpaca-LoRA**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA) political party and is the first left-wing president of Mexico since 1946. He is known for his anti-corruption and anti-neoliberal policies, as well as his commitment to improving the living conditions of the Mexican people.

**Stanford Alpaca**: In 2019, Andres Manuel Lopez Obrador was sworn in as President of Mexico. He is the country’s 58th president and the first from the left-wing Morena party. He was elected in 2018 on a promise to combat corruption and reduce poverty and has faced criticism for his handling of the economy, crime and the country’s relationship with the United States. He is also known for his ambitious social welfare programs and has been praised for raising the minimum wage and providing aid to low-income families. He is seen as a champion of democracy and human rights in Mexico.

**text-davinci-003**: The president of Mexico in 2019 is Andrés Manuel López Obrador, commonly known as AMLO. He is a Mexican politician who has been the President of Mexico since December 1, 2018. He is a member of the National Regeneration Movement (MORENA) party and previously served as the Head of Government of Mexico City from 2000 to 2005. He is a left-wing populist and has promised to reduce corruption and inequality in Mexico. He has also promised to increase public spending on social programs and infrastructure projects.

---

**Instruction**: Tell me about the king of France in 2019.

**Alpaca-LoRA**: The King of France in 2019 is Emmanuel Macron, who was elected in May 2017. He is the youngest president in the history of the Fifth Republic and the first president to be born after World War II. He is also the first president to have never held elected office before.

**Stanford Alpaca**: In 2019, the King of France was Emmanuel Macron. He was sworn in as President of France on May 14, 2017, and has been leading the country ever since. He is a member of the centrist La République En Marche! Party. He is known for his reformist agenda and has faced a number of protests from both the right and left of the political spectrum. He is the youngest French president in history.

**text-davinci-003**: The King of France is a largely symbolic role, as France is a republic. The current President of France is Emmanuel Macron, who was elected in 2017.

---

**Instruction**: List all Canadian provinces in alphabetical order.

**Alpaca-LoRA**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan

**Stanford Alpaca**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, Northwest Territories, Nunavut, Yukon.

**text-davinci-003**:

1. Alberta
2. British Columbia
3. Manitoba
4. New Brunswick
5. Newfoundland and Labrador
6. Northwest Territories
7. Nova Scotia
8. Nunavut
9. Ontario
10. Prince Edward Island
11. Quebec
12. Saskatchewan
13. Yukon

---

**Instruction**: Write a Python program that prints the first 10 Fibonacci numbers.

**Alpaca-LoRA**:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

**Stanford Alpaca**:

```python
numbers = [0, 1]

for i in range(2, 11):
    numbers.append(numbers[i-2] + numbers[i-1])

print(numbers)
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**text-davinci-003**:

```python
# Fibonacci numbers are the numbers in the following integer sequence, called the Fibonacci sequence, and characterized by the fact that every number after the first two is the sum of the two preceding ones.

# Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

# Program:

# Initialize the first two numbers of the Fibonacci sequence
a = 0
b = 1

# Print the first 10 numbers of the Fibonacci sequence
print(a)
print(b)

for i in range(2,10):
    c = a + b
    a = b
    b = c
    print(c)
```

---

**Instruction**: Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.

**Alpaca-LoRA**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

**Stanford Alpaca**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print('FizzBuzz')
    elif i % 3 == 0:
        print('Fizz')
    elif i % 5 == 0:
        print('Buzz')
    else:
        print(i)
```

**text-davinci-003**:

```javascript
for (let i = 1; i <= 100; i++) {
  if (i % 3 === 0 && i % 5 === 0) {
    console.log("FizzBuzz");
  } else if (i % 3 === 0) {
    console.log("Fizz");
  } else if (i % 5 === 0) {
    console.log("Buzz");
  } else {
    console.log(i);
  }
}
```

---

**Instruction**: Tell me five words that rhyme with 'shock'.

**Alpaca-LoRA**: Flock, lock, rock, stock, and sock.

**Stanford Alpaca**: Five words that rhyme with shock are: rock, pop, shock, cook, and snock.

**text-davinci-003**:

1. Block
2. Lock
3. Dock
4. Clock
5. Rock

---

**Instruction**: Translate the sentence 'I have no mouth but I must scream' into Spanish.

**Alpaca-LoRA**: No tengo boca pero tengo que gritar.

**Stanford Alpaca**: No tengo boca, pero debo gritar.

**text-davinci-003**: No tengo boca pero debo gritar.
