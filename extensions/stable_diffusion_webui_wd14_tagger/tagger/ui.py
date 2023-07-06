import os
import json
import gradio as gr

from collections import OrderedDict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

from webui import wrap_gradio_gpu_call
from modules import ui
from modules import generation_parameters_copypaste as parameters_copypaste

from tagger import format, utils
from tagger.utils import split_str
from tagger.interrogator import Interrogator


def unload_interrogators():
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return [f'Successfully unload {unloaded_models} model(s)']

def upload_images(files, output_modelname):
    import shutil
    import requests
    import base64

    dataPath = os.path.join('./data', output_modelname)
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)

    reqFiles = []
    
    for i, f in enumerate(files):
        filename = os.path.basename(f.name)
        saveFilename = os.path.join(dataPath, filename)
        shutil.move(f.name, saveFilename)
        reqFiles.append(('files', (filename, open(saveFilename, 'rb'), 'image/png')))
    
    url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/uploadDreamboothLoraFiles?dreambooth_lora_name="+output_modelname

    auth = 'user:password'
    auth_bytes = auth.encode('UTF-8')
    
    auth_encoded = base64.b64encode(auth_bytes)
    auth_encoded = bytes(auth_encoded)
    auth_encoded_str = auth_encoded.decode('UTF-8')

    headers = {
        'accept': 'application/json',
        'Authorization': 'Basic ' + auth_encoded_str,
    }

    requests.request("POST", url=url, files=reqFiles, headers=headers)

def on_interrogate_req(
    output_modelname: str,
    batch_input_recursive: bool,
    batch_output_action_on_conflict: str,
    batch_remove_duplicated_tag: bool,
    batch_output_save_json: bool,

    interrogator: str,
    threshold: float,
    additional_tags: str,
    exclude_tags: str,
    sort_by_alphabetical_order: bool,
    add_confident_as_weight: bool,
    replace_underscore: bool,
    replace_underscore_excludes: str,
    escape_tag: bool,

    unload_model_after_running: bool
):
    import requests
    import base64

    url = "http://mwgpu.mydomain.blog:4000/tagger/v1/DataInterrogate"

    auth = 'user:password'
    auth_bytes = auth.encode('UTF-8')

    auth_encoded = base64.b64encode(auth_bytes)
    auth_encoded = bytes(auth_encoded)
    auth_encoded_str = auth_encoded.decode('UTF-8')

    from extensions.stable_diffusion_webui_wd14_tagger.tagger.api_models import DataInterrogateRequest

    req = DataInterrogateRequest(
        output_modelname = output_modelname,
        batch_input_recursive= batch_input_recursive,
        batch_output_action_on_conflict= batch_output_action_on_conflict,
        batch_remove_duplicated_tag= batch_remove_duplicated_tag,
        batch_output_save_json= batch_output_save_json,
        interrogator= interrogator,
        threshold= threshold,
        additional_tags= additional_tags,
        exclude_tags= exclude_tags,
        sort_by_alphabetical_order= sort_by_alphabetical_order,
        add_confident_as_weight= add_confident_as_weight,
        replace_underscore= replace_underscore,
        replace_underscore_excludes= replace_underscore_excludes,
        escape_tag= escape_tag,
        unload_model_after_running= unload_model_after_running,
    )

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + auth_encoded_str
    }

    response = requests.request("POST", url=url, headers=headers, data=req.json())
    print(response.json())

def on_interrogate(
    output_modelname: str,
    batch_input_recursive: bool,
    batch_output_action_on_conflict: str,
    batch_remove_duplicated_tag: bool,
    batch_output_save_json: bool,

    interrogator: str,
    threshold: float,
    additional_tags: str,
    exclude_tags: str,
    sort_by_alphabetical_order: bool,
    add_confident_as_weight: bool,
    replace_underscore: bool,
    replace_underscore_excludes: str,
    escape_tag: bool,

    unload_model_after_running: bool
):
    utils.refresh_interrogators()
    if interrogator not in utils.interrogators:
        return ['', None, None, f"'{interrogator}' is not a valid interrogator"]

    interrogator: Interrogator = utils.interrogators[interrogator]

    postprocess_opts = (
        threshold,
        split_str(additional_tags),
        split_str(exclude_tags),
        sort_by_alphabetical_order,
        add_confident_as_weight,
        replace_underscore,
        split_str(replace_underscore_excludes),
        escape_tag
    )
    
    supported_extensions = [
        e for e, f in Image.registered_extensions().items() if f in Image.OPEN
    ]

    base_dir = os.path.join('./data/dreambooth_lora', output_modelname, 'images')

    paths = glob(os.path.join(base_dir, '*.png'))
    

    output_paths = []
    for path in paths:
        try:
            image = Image.open(path)
        except UnidentifiedImageError:
            # just in case, user has mysterious file...
            print(f'${path} is not supported image type')
            continue
        
        output_filename = Path(path).parts[-1]
        output_filename = output_filename.replace('.png', '.txt')

        output_dir = base_dir

        output_path = os.path.join(output_dir, output_filename)

        output_path = Path(output_path)

        output = []

        ratings, tags = interrogator.interrogate(image)
        processed_tags = Interrogator.postprocess_tags(tags, *postprocess_opts)

        plain_tags = ', '.join(processed_tags)

        if batch_output_action_on_conflict == 'copy':
            output = [plain_tags]
        elif batch_output_action_on_conflict == 'prepend':
            output.insert(0, plain_tags)
        else:
            output.append(plain_tags)

        if batch_remove_duplicated_tag:
            output_path.write_text(
                ', '.join(
                    OrderedDict.fromkeys(
                        map(str.strip, ','.join(output).split(','))
                    )
                ),
                encoding='utf-8'
            )
        else:
            output_path.write_text(
                ', '.join(output),
                encoding='utf-8'
            )
        # if output_path.is_file():
        #     output_paths.append(output_path)


        

    return ', '.join(output_paths)

    # batch process
    # batch_input_glob = batch_input_glob.strip()
    # batch_output_dir = batch_output_dir.strip()
    # batch_output_filename_format = batch_output_filename_format.strip()
    
    # if batch_input_glob != '':
    #     # if there is no glob pattern, insert it automatically
    #     if not batch_input_glob.endswith('*'):
    #         if not batch_input_glob.endswith(os.sep):
    #             batch_input_glob += os.sep
    #         batch_input_glob += '*'

    #     # get root directory of input glob pattern
    #     base_dir = batch_input_glob.replace('?', '*')
    #     base_dir = base_dir.split(os.sep + '*').pop(0)

    #     # check the input directory path
    #     if not os.path.isdir(base_dir):
    #         return ['', None, None, 'input path is not a directory']

    #     # this line is moved here because some reason
    #     # PIL.Image.registered_extensions() returns only PNG if you call too early
    #     supported_extensions = [
    #         e
    #         for e, f in Image.registered_extensions().items()
    #         if f in Image.OPEN
    #     ]

    #     paths = [
    #         Path(p)
    #         for p in glob(batch_input_glob, recursive=batch_input_recursive)
    #         if '.' + p.split('.').pop().lower() in supported_extensions
    #     ]

    #     print(f'found {len(paths)} image(s)')

    #     for path in paths:
    #         try:
    #             image = Image.open(path)
    #         except UnidentifiedImageError:
    #             # just in case, user has mysterious file...
    #             print(f'${path} is not supported image type')
    #             continue

    #         # guess the output path
    #         base_dir_last = Path(base_dir).parts[-1]
    #         base_dir_last_idx = path.parts.index(base_dir_last)
    #         output_dir = Path(
    #             batch_output_dir) if batch_output_dir else Path(base_dir)
    #         output_dir = output_dir.joinpath(
    #             *path.parts[base_dir_last_idx + 1:]).parent

    #         output_dir.mkdir(0o777, True, True)

    #         # format output filename
    #         format_info = format.Info(path, 'txt')

    #         try:
    #             formatted_output_filename = format.pattern.sub(
    #                 lambda m: format.format(m, format_info),
    #                 batch_output_filename_format
    #             )
    #         except (TypeError, ValueError) as error:
    #             return ['', None, None, str(error)]

    #         output_path = output_dir.joinpath(
    #             formatted_output_filename
    #         )

    #         output = []

    #         if output_path.is_file():
    #             output.append(output_path.read_text(errors='ignore').strip())

    #             if batch_output_action_on_conflict == 'ignore':
    #                 print(f'skipping {path}')
    #                 continue

    #         ratings, tags = interrogator.interrogate(image)
    #         processed_tags = Interrogator.postprocess_tags(
    #             tags,
    #             *postprocess_opts
    #         )

    #         # TODO: switch for less print
    #         print(
    #             f'found {len(processed_tags)} tags out of {len(tags)} from {path}'
    #         )

    #         plain_tags = ', '.join(processed_tags)

    #         if batch_output_action_on_conflict == 'copy':
    #             output = [plain_tags]
    #         elif batch_output_action_on_conflict == 'prepend':
    #             output.insert(0, plain_tags)
    #         else:
    #             output.append(plain_tags)

    #         if batch_remove_duplicated_tag:
    #             output_path.write_text(
    #                 ', '.join(
    #                     OrderedDict.fromkeys(
    #                         map(str.strip, ','.join(output).split(','))
    #                     )
    #                 ),
    #                 encoding='utf-8'
    #             )
    #         else:
    #             output_path.write_text(
    #                 ', '.join(output),
    #                 encoding='utf-8'
    #             )

    #         if batch_output_save_json:
    #             output_path.with_suffix('.json').write_text(
    #                 json.dumps([ratings, tags])
    #             )

    #     print('all done :)')

    # if unload_model_after_running:
    #     interrogator.unload()

    # return ['', None, None, '']


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                # input components
                with gr.Tabs():
                    with gr.TabItem(label="Load Images"):
                        files_upload = gr.File(file_count="multiple")

                        batch_input_recursive = utils.preset.component(
                            gr.Checkbox,
                            label='Use recursive with glob pattern'
                        )

                        output_modelname = utils.preset.component(
                            gr.Textbox,
                            label='Output model name',
                        )

                        batch_output_action_on_conflict = utils.preset.component(
                            gr.Dropdown,
                            label='Action on existing caption',
                            value='ignore',
                            choices=[
                                'ignore',
                                'copy',
                                'append',
                                'prepend'
                            ]
                        )

                        batch_remove_duplicated_tag = utils.preset.component(
                            gr.Checkbox,
                            label='Remove duplicated tag'
                        )

                        batch_output_save_json = utils.preset.component(
                            gr.Checkbox,
                            label='Save with JSON'
                        )

                submit = gr.Button(
                    value='Interrogate',
                    variant='primary'
                )

                info = gr.HTML()

                # preset selector
                with gr.Row(variant='compact'):
                    available_presets = utils.preset.list()
                    selected_preset = gr.Dropdown(
                        label='Preset',
                        choices=available_presets,
                        value=available_presets[0]
                    )

                    save_preset_button = gr.Button(
                        value=ui.save_style_symbol
                    )

                    ui.create_refresh_button(
                        selected_preset,
                        lambda: None,
                        lambda: {'choices': utils.preset.list()},
                        'refresh_preset'
                    )

                # option components

                # interrogator selector
                with gr.Column():
                    with gr.Row(variant='compact'):
                        interrogator_names = utils.refresh_interrogators()
                        interrogator = utils.preset.component(
                            gr.Dropdown,
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )

                        ui.create_refresh_button(
                            interrogator,
                            lambda: None,
                            lambda: {'choices': utils.refresh_interrogators()},
                            'refresh_interrogator'
                        )

                    unload_all_models = gr.Button(
                        value='Unload all interrogate models'
                    )

                threshold = utils.preset.component(
                    gr.Slider,
                    label='Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )

                additional_tags = utils.preset.component(
                    gr.Textbox,
                    label='Additional tags (split by comma)',
                    elem_id='additioanl-tags'
                )

                exclude_tags = utils.preset.component(
                    gr.Textbox,
                    label='Exclude tags (split by comma)',
                    elem_id='exclude-tags'
                )

                sort_by_alphabetical_order = utils.preset.component(
                    gr.Checkbox,
                    label='Sort by alphabetical order',
                )
                add_confident_as_weight = utils.preset.component(
                    gr.Checkbox,
                    label='Include confident of tags matches in results'
                )
                replace_underscore = utils.preset.component(
                    gr.Checkbox,
                    label='Use spaces instead of underscore',
                    value=True
                )
                replace_underscore_excludes = utils.preset.component(
                    gr.Textbox,
                    label='Excudes (split by comma)',
                    # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
                    value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                )
                escape_tag = utils.preset.component(
                    gr.Checkbox,
                    label='Escape brackets',
                )

                unload_model_after_running = utils.preset.component(
                    gr.Checkbox,
                    label='Unload model after running',
                )

            # output components
            with gr.Column(variant='panel'):
                tags = gr.Textbox(
                    label='Tags',
                    placeholder='Found tags',
                    interactive=False
                )

                with gr.Row():
                    parameters_copypaste.bind_buttons(
                        parameters_copypaste.create_buttons(
                            ["txt2img", "img2img"],
                        ),
                        None,
                        tags
                    )

                rating_confidents = gr.Label(
                    label='Rating confidents',
                    elem_id='rating-confidents'
                )
                tag_confidents = gr.Label(
                    label='Tag confidents',
                    elem_id='tag-confidents'
                )

        # register events
        selected_preset.change(
            fn=utils.preset.apply,
            inputs=[selected_preset],
            outputs=[*utils.preset.components, info]
        )

        save_preset_button.click(
            fn=utils.preset.save,
            inputs=[selected_preset, *utils.preset.components],  # values only
            outputs=[info]
        )

        unload_all_models.click(
            fn=unload_interrogators,
            outputs=[info]
        )

        files_upload.upload(
            fn=upload_images,
            inputs=[files_upload, output_modelname],
            show_progress=True
        )

        # files_upload.upload(
        #     fn=modules.textual_inversion.ui.uploadFiles,
        #     inputs=[files_upload, train_embedding_name],
        #     show_progress=True,
        # )

        submit.click(
            fn= on_interrogate_req,
            inputs=[
                    # single process
                    # image,

                    # batch process
                    # batch_input_glob,
                    
                    output_modelname,

                    batch_input_recursive,
                    # batch_output_dir,
                    # batch_output_filename_format,
                    batch_output_action_on_conflict,
                    batch_remove_duplicated_tag,
                    batch_output_save_json,

                    # options
                    interrogator,
                    threshold,
                    additional_tags,
                    exclude_tags,
                    sort_by_alphabetical_order,
                    add_confident_as_weight,
                    replace_underscore,
                    replace_underscore_excludes,
                    escape_tag,

                    unload_model_after_running

            ],
            # outputs=[
            #     tags,
            #     rating_confidents,
            #     tag_confidents,
            #     info
            # ]
        )

    return [(tagger_interface, "Tagger", "tagger")]
