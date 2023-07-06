import gradio as gr
import os
import argparse
from dreambooth_gui import dreambooth_tab
from finetune_gui import finetune_tab
from textual_inversion_gui import ti_tab
from library.utilities import utilities_tab
from library.extract_lora_gui import gradio_extract_lora_tab
from library.extract_lycoris_locon_gui import gradio_extract_lycoris_locon_tab
from library.merge_lora_gui import gradio_merge_lora_tab
from library.resize_lora_gui import gradio_resize_lora_tab
from library.extract_lora_from_dylora_gui import gradio_extract_dylora_tab
from library.merge_lycoris_gui import gradio_merge_lycoris_tab
from lora_gui import lora_tab

import os
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

def UI():
    headless = False #kwargs.get('headless', False)
    interface = gr.Blocks( title=f'Kohya_ss GUI', theme=gr.themes.Default()
    )

    with interface:
        with gr.Tab('Dreambooth'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(headless=headless)
        with gr.Tab('Dreambooth LoRA'):
            lora_tab(headless=headless)
        with gr.Tab('Dreambooth TI'):
            ti_tab(headless=headless)
        with gr.Tab('Finetune'):
            finetune_tab(headless=headless)
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )
            with gr.Tab('LoRA'):
                gradio_extract_dylora_tab(headless=headless)
                gradio_extract_lora_tab(headless=headless)
                gradio_extract_lycoris_locon_tab(headless=headless)
                gradio_merge_lora_tab(headless=headless)
                gradio_merge_lycoris_tab(headless=headless)
                gradio_resize_lora_tab(headless=headless)
    
    return [(interface, "Kohya", "kohya")]