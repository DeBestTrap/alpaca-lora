# %%
import sys
import torch
from peft import PeftModel
import transformers
import gradio as gr

# from transformers import GPT2Tokenizer as Tokenizer
# from transformers import GPT2LMHeadModel as Model
from transformers import AutoModelForCausalLM as Model
from transformers import AutoTokenizer as Tokenizer
from transformers import GenerationConfig 
# from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


LOAD_8BIT = False
# BASE_MODEL = "gpt2-xl"
# LORA_WEIGHTS = "./lora/gpt2-mod/"
BASE_MODEL = "EleutherAI/gpt-j-6B"
LORA_WEIGHTS = "lora/gptj-chip2"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


# %%
if device == "cuda":
    model = Model.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        # revision="float16",
        revision="sharded",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
        device_map="auto"
    )
elif device == "mps":
    model = Model.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = Model.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )

# %%
# tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer = Tokenizer.from_pretrained(BASE_MODEL)

# %%
print(tokenizer.tokenize("\n\n"))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("\n\n")))
# %% Functions
def generate_prompt(instruction, input=None):
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

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
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
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
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
    # return output.split("### Response:")[1].strip()
    return output.split("### Response:")[-1].split("\n\n")[0].strip() + "\n\n" + output

# %%

gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1, value=0.8, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=4, step=1, value=3, label="Beams"),
        gr.components.Slider(minimum=1, maximum=2, step=0.05, value=1.2, label="Repetition Penalty"),
        gr.components.Slider(minimum=-3, maximum=3, step=0.05, value=1, label="Length Penalty"),
        gr.components.Slider(minimum=0, maximum=10, step=1, value=0, label="ngram size"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="🦙🌲 Alpaca-LoRA",
    description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
).launch()

# Old testing code follows.

"""
if __name__ == "__main__":
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
"""

# %%
