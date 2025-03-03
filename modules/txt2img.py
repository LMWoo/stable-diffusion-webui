import modules.scripts
from modules import sd_samplers, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html



def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_sampler_index: int, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args):
    override_settings = create_override_settings_dict(override_settings_texts)

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_sampler_name=sd_samplers.samplers_for_img2img[hr_sampler_index - 1].name if hr_sampler_index != 0 else None,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is None:
        processed = processing.process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)

def txt2imgReq(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_sampler_index: int, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args):
    import requests
    import json
    import base64
    from modules.api.models import TextToImageResponse
    from modules.api.api import decode_base64_to_image

    url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/txt2img"

    payload = json.dumps({
    "enable_hr": False,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "string",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_sampler_name": "string",
    "hr_prompt": "",
    "hr_negative_prompt": "",
    "prompt": prompt,
    "styles": [
        "string"
    ],
    "seed": seed,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "Euler a",
    "batch_size": batch_size,
    "n_iter": n_iter,
    "steps": steps,
    "cfg_scale": cfg_scale,
    "width": width,
    "height": height,
    "restore_faces": restore_faces,
    "tiling": tiling,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "negative_prompt": negative_prompt,
    "eta": 0,
    "s_min_uncond": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings": {"sd_model_checkpoint": shared.opts.data['sd_model_checkpoint'],},
    "override_settings_restore_afterwards": True,
    "script_args": [],
    "sampler_index": sampler_index,
    "script_name": "",
    "send_images": True,
    "save_images": False,
    "alwayson_scripts": {}
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
    res = TextToImageResponse(**(response.json()))

    img = decode_base64_to_image(res.images[0])
    return [img], "", "", ""