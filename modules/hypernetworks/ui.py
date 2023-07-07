import html

import gradio as gr
import modules.hypernetworks.hypernetwork
from modules import devices, sd_hijack, shared

not_available = ["hardswish", "multiheadattention"]
keys = [x for x in modules.hypernetworks.hypernetwork.HypernetworkModule.activation_dict if x not in not_available]

def uploadHypernetworkFilesReq(files, train_hypernetwork_name):
    import os
    import shutil
    import base64
    import requests

    reqFiles = []

    if not os.path.exists('./data'):
        os.mkdir('./data')

    for i, f in enumerate(files):
        filename = os.path.basename(f.name)
        saveFilename = os.path.join("./data", filename)
        shutil.move(f.name, saveFilename)
        reqFiles.append(('files', (filename, open(saveFilename, 'rb'), 'image/png')))

    url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/uploadHypernetworkFiles?hypernetwork_name="+train_hypernetwork_name    
    
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


def create_hypernetwork_req(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None):
    import requests
    import json
    import base64

    url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/create/hypernetwork"

    payload = json.dumps({
        "name": name,
        "enable_sizes": enable_sizes,
        "overwrite_old": overwrite_old,
        "layer_structure": layer_structure,
        "activation_func": activation_func,
        "weight_init": weight_init,
        "add_layer_norm": add_layer_norm,
        "use_dropout": use_dropout,
        "dropout_structure": dropout_structure,
    })

    auth = 'user:password'
    auth_bytes = auth.encode('UTF-8')

    auth_encoded = base64.b64encode(auth_bytes)
    auth_encoded = bytes(auth_encoded)
    auth_encoded_str = auth_encoded.decode('UTF-8')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + auth_encoded_str
    }

    response = requests.request("POST", url=url, headers=headers, data=payload)

    return name, f"Created Hypernetwork: {name}", ""

def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None):
    filename = modules.hypernetworks.hypernetwork.create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure, activation_func, weight_init, add_layer_norm, use_dropout, dropout_structure)

    return gr.Dropdown.update(choices=sorted(shared.hypernetworks)), f"Created HyperNetwork: {name}", ""


def train_hypernetwork(*args):
    shared.loaded_hypernetworks = []

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram is not possible'

    try:
        sd_hijack.undo_optimizations()

        hypernetwork, filename = modules.hypernetworks.hypernetwork.train_hypernetwork(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {hypernetwork.step} steps.
Hypernetwork saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        shared.sd_model.cond_stage_model.to(devices.device)
        shared.sd_model.first_stage_model.to(devices.device)
        sd_hijack.apply_optimizations()

