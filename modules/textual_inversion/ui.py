import html

import gradio as gr

import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import sd_hijack, shared

def uploadFiles(files):
    import os
    import shutil
    for i, f in enumerate(files):
        filename = os.path.basename(f.name)

        shutil.move(f.name, os.path.join("./modules/textual_inversion/images", filename))

def create_embedding(name, initialization_text, nvpt, overwrite_old):
    import requests
    import json
    import base64
    from modules.api.models import PreprocessResponse

    url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/create/embedding"

    payload = json.dumps({
        "name": name,
        "init_text": initialization_text,
        "num_vectors_per_token": nvpt,
        "overwrite_old": overwrite_old,
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
    print(response.json())


    return None, f"Created embedding", ""
    # filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, overwrite_old, init_text=initialization_text)

    # sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    # return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    modules.textual_inversion.preprocess.preprocess(*args)

    return f"Preprocessing {'interrupted' if shared.state.interrupted else 'finished'}.", ""


def train_embedding(*args):

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    apply_optimizations = shared.opts.training_xattention_optimizations
    try:
        if not apply_optimizations:
            sd_hijack.undo_optimizations()

        embedding, filename = modules.textual_inversion.textual_inversion.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        if not apply_optimizations:
            sd_hijack.apply_optimizations()

