# %%
import os
import gradio as gr
from typing import List, Union, Dict
import util
import json
from util import SmarterSet


def create_model_for_generating(version: str, revision: str, load_8bit: str,
                                lora: str, verbose:bool=False, progress=gr.Progress(track_tqdm=True)):
    util.create_model(version, revision, load_8bit, verbose)
    return util.prepare_model_for_generating(load_8bit, lora, verbose)


def create_model_for_finetuning(model: str, version: str, revision: str,
                                load_8bit: str, model_choices: dict,
                                verbose:bool=False,
                                progress=gr.Progress(track_tqdm=True)):
    if model in model_choices.keys():
        target_modules = model_choices[model]["target_modules"]
    else:
        target_modules = ""
    util.create_model(version, revision, load_8bit, verbose)
    return util.prepare_model_for_training(target_modules, verbose)


def change_model_dropdown_choices(model_choices: dict, nested_key: str,
                                  choice: str):
    # TODO maybe fix? typing.Dict causes issues when used here: Dict[str, Dict[str, List[str]]]
    if choice in model_choices.keys():
        return change_dropdown_choices(model_choices[choice][nested_key])
    else:
        return change_dropdown_choices([])


def change_dropdown_choices(l: List[str]):
    return gr.update(choices=l, value="")


def show_lora_checkpoints(choice):
    l = []
    if choice != "New Lora":
        for name in os.listdir(os.path.join(os.getcwd(), "lora/", choice)):
            if os.path.isdir(os.path.join(os.getcwd(), "lora/", choice, name)):
                l.append(name)
    return change_dropdown_choices(l)

    #         lambda choice: change_dropdown_choices(
    # [name for name in os.listdir(os.path.join(os.getcwd(), "lora/", choice))
    #             if os.path.isdir(os.path.join(os.getcwd(), "lora/", choice, name))]
    # if choice != "New Lora" else []),


def recent_models():
    try:
        with open("recent_models_cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def cache_recent_model_choices(model: str,
                               version: str,
                               revision: str,
                               load_8bit: str,
                               lora: str):
    cache = recent_models()
    if len(cache) > 5:
        cache.pop(0)
    list = [
        f"{lora}" if lora else f"{version}, {revision}",
        {
            "model": model,
            "version": version,
            "revision": revision,
            "load_8bit": load_8bit,
            "lora": lora,
        }
    ]

    if not list in cache:
        cache.append(list)
    with open("recent_models_cache.json", "w") as f:
        json.dump(cache, f)
    return cache


def change_dropdowns_to_recent_model(idx):
    choice = recent_models()[idx][1]
    return [
        gr.Dropdown.update(value=choice["model"]),
        gr.Dropdown.update(value=choice["version"]),
        gr.Dropdown.update(value=choice["revision"]),
        gr.Radio.update(value=choice["load_8bit"]),
        gr.Dropdown.update(value=choice["lora"])
    ]


def show_model_options_generate():
    # TODO Add Hide Lora Config
    '''
    Updates the accordion header to "Load Model For Generating".
    Hides checkpoints dropdown, and load for finetuning button.
    Reveals load for generating button.
    '''
    return [
        gr.Accordion.update(label="Load Model For Generating"),
        gr.Dropdown.update(visible=False),
        gr.Button.update(visible=False),
        gr.Button.update(visible=True),
        gr.Dropdown.update(
            choices=os.listdir(os.path.join(os.getcwd(), "lora/")))
    ]


def show_model_options_finetune():
    # TODO Add show Lora Config
    '''
    Updates the accordion header to "Load Model For Finetuning".
    Hides load for generating button
    Reveals checkpoints dropdown, and load for finetuning button
    '''
    return [
        gr.Accordion.update(label="Load Model For Finetuning"),
        gr.Button.update(visible=False),
        gr.Dropdown.update(visible=True),
        gr.Button.update(visible=True),
        gr.Dropdown.update(choices=["New Lora"] +
                           os.listdir(os.path.join(os.getcwd(), "lora/")))
    ]


with gr.Blocks() as interface:
    # CONSTANT gr states 
    STATE_MODEL_CHOICES_DICT = gr.State(value=util.load_models_json())
    STATE_VERSION_STR = gr.State(value="version")
    STATE_REVISION_STR = gr.State(value="revision")
    STATE_TRUE = gr.State(value=True)
    STATE_FALSE = gr.State(value=False)

    button_verbose = gr.Checkbox(label="Verbose", interactive=True)
    with gr.Accordion("Load Model For Generating") as accordion_load_model:
        with gr.Column():
            text_loaded_model = gr.Textbox(
                show_label=False,
                value="No Model Loaded",
                interactive=False,
            ).style(container=False)
            with gr.Row(variant="panel"):
                dropdown_model = gr.Dropdown(
                    choices=list(STATE_MODEL_CHOICES_DICT.value.keys()),
                    label="Model",
                    interactive=True).style(container=False)
                dropdown_version = gr.Dropdown(
                    choices=[], label="Version",
                    interactive=True).style(container=False)
                dropdown_revision = gr.Dropdown(
                    choices=[], label="Revision",
                    interactive=True).style(container=False)
                radio_load_8bit_model = gr.Radio(choices=["True", "False"],
                                                 label="Load 8bit Model",
                                                 interactive=True)
            with gr.Row(variant="panel"):
                dropdown_lora = gr.Dropdown(
                    choices=os.listdir(os.path.join(os.getcwd(), "lora/")),
                    label="Lora",
                    interactive=True).style(container=False)
                dropdown_checkpoint = gr.Dropdown(
                    choices=[],
                    label="Lora Checkpoint",
                    visible=False,
                    interactive=True).style(container=False)
                # TODO Add LORA config here
            with gr.Row(variant="panel"):
                dataset_recent_models = gr.Dataset(
                    components=[gr.Dropdown(visible=False)],
                    label="Recently Loaded Models",
                    samples=recent_models(),
                    type="index")
            with gr.Accordion("FAQ", open=False):
                gr.Markdown("""
                ### What is the difference between version and revision?
                Some models may have different versions with different number of parameters 
                - i.e. a 1.5 Billion parameter and a 7 Billion one.
                Revisions of a model keep the same number of paramters but the way that the data is
                stored is different usually.

                So when you choose gpt2 theres different sizes of models like S, XL

                """)
            button_load_model_g = gr.Button(variant="primary",
                                            value="Load Model For Generating",
                                            interactive=True)
            button_load_model_f = gr.Button(variant="primary",
                                            value="Load Model For Finetuning",
                                            visible=False,
                                            interactive=True)

            dropdown_model.select(
                change_model_dropdown_choices,
                inputs=[
                    STATE_MODEL_CHOICES_DICT, STATE_VERSION_STR, dropdown_model
                ],
                outputs=dropdown_version,
            )
            dropdown_model.select(
                change_model_dropdown_choices,
                inputs=[
                    STATE_MODEL_CHOICES_DICT, STATE_REVISION_STR,
                    dropdown_model
                ],
                outputs=dropdown_revision,
            )
            dropdown_lora.change(
                show_lora_checkpoints,
                inputs=dropdown_lora,
                outputs=dropdown_checkpoint,
            )

            dataset_recent_models.click(change_dropdowns_to_recent_model,
                                        inputs=dataset_recent_models,
                                        outputs=[
                                            dropdown_model,
                                            dropdown_version,
                                            dropdown_revision,
                                            radio_load_8bit_model,
                                            dropdown_lora
                                        ])
            button_load_model_g.click(create_model_for_generating,
                                      show_progress=True,
                                      inputs=[
                                          dropdown_version, dropdown_revision,
                                          radio_load_8bit_model, dropdown_lora,
                                          button_verbose,
                                      ],
                                      outputs=text_loaded_model)
            button_load_model_f.click(create_model_for_finetuning,
                                      show_progress=True,
                                      inputs=[
                                          dropdown_model, dropdown_version,
                                          dropdown_revision,
                                          radio_load_8bit_model,
                                          STATE_MODEL_CHOICES_DICT,
                                          button_verbose
                                      ],
                                      outputs=text_loaded_model)
            button_load_model_g.click(cache_recent_model_choices,
                                      inputs=[
                                          dropdown_model, dropdown_version,
                                          dropdown_revision,
                                          radio_load_8bit_model, dropdown_lora
                                      ],
                                      outputs=dataset_recent_models)
            button_load_model_f.click(cache_recent_model_choices,
                                      inputs=[
                                          dropdown_model, dropdown_version,
                                          dropdown_revision,
                                          radio_load_8bit_model, dropdown_lora
                                      ],
                                      outputs=dataset_recent_models)

    with gr.Tab("Generate/Evaluate") as tab_generate:
        tab_generate.select(show_model_options_generate,
                            outputs=[
                                accordion_load_model, dropdown_checkpoint,
                                button_load_model_f, button_load_model_g,
                                dropdown_lora
                            ])
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():  # GENERATION PARAMETERS
                    text_instruction = gr.Textbox(
                        lines=2,
                        label="Instruction",
                        placeholder="Tell me about alpacas.")
                    text_input = gr.Textbox(lines=2,
                                            label="Input",
                                            placeholder="none")
                    slider_temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.8,
                        label="Temperature",
                        interactive=True,
                    )
                    slider_top_p = gr.Slider(minimum=0,
                                             maximum=1,
                                             value=0.75,
                                             label="Top p",
                                             interactive=True)
                    slider_top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=40,
                        label="Top k",
                        interactive=True,
                    )
                    slider_beams = gr.Slider(
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=3,
                        label="Beams",
                        interactive=True,
                    )
                    slider_repetition_penalty = gr.Slider(
                        minimum=1,
                        maximum=2,
                        step=0.05,
                        value=1.2,
                        label="Repetition Penalty",
                        interactive=True,
                    )
                    slider_length_penalty = gr.Slider(
                        minimum=-3,
                        maximum=3,
                        step=0.05,
                        value=1,
                        label="Length Penalty",
                        interactive=True,
                    )
                    slider_ngram_size = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                        label="ngram size",
                        interactive=True,
                    )
                    slider_max_tokens = gr.Slider(
                        minimum=1,
                        maximum=2000,
                        step=1,
                        value=128,
                        label="Max tokens",
                        interactive=True,
                    )
                with gr.Column():
                    button_g = gr.Button(value="Generate",
                                         variant="primary",
                                         interactive=True)
                    output_g = gr.Textbox(lines=5,
                                          label="Output",
                                          interactive=False)
                                          
                    button_g.click(
                        util.generate,
                        inputs=[
                            text_instruction, text_input, slider_temperature,
                            slider_top_p, slider_top_k, slider_beams,
                            slider_repetition_penalty, slider_length_penalty,
                            slider_ngram_size, slider_max_tokens, STATE_TRUE,
                            button_verbose,
                        ],
                        outputs=[output_g])
        with gr.Tab("Evaluate"):
            with gr.Column(variant="panel"):
                state_selected_data = gr.State(value=SmarterSet())
                dropdown_data_files_e = gr.Dropdown(
                    choices=os.listdir(os.path.join(os.getcwd(), "data/")),
                    label="Data",
                    interactive=True).style(container=False)
                text_ = gr.Textbox(label="Data files to evaluate",
                                   value="{}",
                                   interactive=False)
                with gr.Row():
                    button_add_data = gr.Button(value="Add",
                                                variant="secondary",
                                                interactive=True)
                    button_clear_data = gr.Button(value="Clear",
                                                  variant="secondary",
                                                  interactive=True)
                    # TODO make output prettier
            button_e = gr.Button(value="Evaluate",
                                 variant="primary",
                                 interactive=True)
            json_output = gr.Json()

            button_add_data.click(
                lambda set, ele: [set.add(ele),
                                  gr.update(value=set)],
                inputs=[state_selected_data, dropdown_data_files_e],
                outputs=[state_selected_data, text_])
            button_clear_data.click(
                lambda set: [set.clear(), gr.update(value=set)],
                inputs=state_selected_data,
                outputs=[state_selected_data, text_])
            button_e.click(util.evaluate_model,
                           inputs=[
                               state_selected_data, radio_load_8bit_model,
                               dropdown_lora, dropdown_version,
                               button_verbose
                           ],
                           outputs=json_output)
        with gr.Tab("Boolq"):
            text_output_boolq = gr.Textbox("Select a prompt style and generate responses", show_label=False, interactive=False)
            radio_boolq_prompt_styles = gr.Radio(["Zero-Shot", "One-Shot", "Few-Shot"], value="Few-Shot", label="Prompt Style")
            button_run_boolq = gr.Button("Generate Responses for Boolq")
            button_evaluate_boolq = gr.Button("Evaluate Responses for Boolq")
            button_run_boolq.click(util.generate_boolq_responses, inputs=radio_boolq_prompt_styles, outputs=text_output_boolq)

    with gr.Tab("Finetune") as tab_finetune:
        tab_finetune.select(show_model_options_finetune,
                            outputs=[
                                accordion_load_model, button_load_model_g,
                                dropdown_checkpoint, button_load_model_f,
                                dropdown_lora
                            ])
        with gr.Row():
            with gr.Column():  # FINETUNING HYPERPARAMETERS
                slider_batch_size = gr.Slider(
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=128,
                    label="Batch Size",
                    interactive=True,
                )
                slider_micro_batch_size = gr.Slider(
                    minimum=1,
                    maximum=24,
                    step=1,
                    value=4,
                    label="Micro Batch Size",
                    interactive=True,
                )
                slider_save_eval_steps = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    step=1,
                    value=200,
                    label="Steps until Save and Evaluation",
                    interactive=True,
                )
                slider_epochs = gr.Slider(
                    minimum=0,
                    maximum=4,
                    step=0.05,
                    value=1,
                    label="Epochs",
                    interactive=True,
                )
                slider_learning_rate = gr.Slider(
                    minimum=3e-5,
                    maximum=3e-3,
                    step=3e-5,
                    value=3e-4,
                    label="Learning Rate",
                    interactive=True,
                )
                slide_val_set_size = gr.Slider(
                    minimum=0,
                    maximum=4000,
                    step=1,
                    value=2000,
                    label="Validation Set Size",
                    interactive=True,
                )
            with gr.Column():
                dropdown_data_path_f = gr.Dropdown(
                    choices=os.listdir(os.path.join(os.getcwd(), "data/")),
                    label="Data",
                    interactive=True).style(container=False)
                button_f = gr.Button(value="Finetune",
                                     variant="primary",
                                     interactive=True)
                output_f = gr.Textbox(interactive=False)

                button_f.click(util.train,
                               inputs=[
                                   dropdown_data_path_f, dropdown_lora,
                                   dropdown_checkpoint, slider_batch_size,
                                   slider_micro_batch_size,
                                   slider_save_eval_steps, slider_epochs,
                                   slider_learning_rate, slide_val_set_size,
                                   button_verbose
                               ],
                               outputs=[output_f])

    with gr.Accordion("Known Bugs"):
        gr.Markdown("""
                    ### After creating model for generating or evaluating, trying to finetune may cause issues (vice versa also).
                    Solution: reload model

                    ### Using a recently loaded model will not show dropdowns for version and revision.
                    Solution: Load the dropdowns choices by clicking a Model in the Model dropdown.

                    ### ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported.
                    Solution: Find the LLaMA tokenizer config in your huggingface cache and change the capitilization from LLaMATokenizer to LlamaTokenizer.
                    ```
                    WINDOWS: C:\\Users\\USERSNAME\\.cache\\huggingface\\hub\\models--decapoda-research--llama-7b-hf\\snapshots\\HASH\\tokenizer_config.json
                    LINUX: ~/.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/HASH/tokenizer_config.json
                    ```

                    """)

interface.queue(concurrency_count=2).launch()

# # %%
# utils.prepare_model_for_training()
# # %%
# utils.load_lora_weights("lora/gpt2-mod")
# # %%
# # Blocks Example

# with gr.Blocks() as demo:
#     radio = gr.Dropdown([1, 2, 4], label="Set the value of the number")
#     number = gr.Dropdown([0], value=0, interactive=True)
#     radio.change(fn=lambda c: gr.update(choices=[c]),
#                  inputs=radio,
#                  outputs=number)
# demo.launch()
# # %%
# import time

# import gradio as gr
# from tqdm import tqdm

# def foo(pr=gr.Progress(track_tqdm=True)):
#     for i in tqdm(range(10)):
#         print(i)
#         time.sleep(1)

# with gr.Blocks() as demo:
#     btn = gr.Button(label="Sleep")
#     out = gr.Textbox(interactive=False)
#     btn.click(fn=foo, outputs=out)

# demo.queue().launch()
# %%

# %%

# %%
