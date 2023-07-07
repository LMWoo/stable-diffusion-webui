from typing import Callable, Any
from threading import Lock
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from secrets import compare_digest
from pydantic import BaseModel, Field
from modules.call_queue import queue_lock

from modules import shared

class TutorialRequest(BaseModel):
    data: str = Field(title='data', description="tutorial data request")

class TutorialResponse(BaseModel):
    data: str = Field(title='data', description="tutorial data response")

class DreamboothLoraFolderPreparationRequest(BaseModel):
    util_output_model_name : str
    util_training_images_dir_input : Any
    util_training_images_repeat_input : Any
    util_instance_prompt_input : Any
    util_regularization_images_dir_input : Any
    util_regularization_images_repeat_input : Any
    util_class_prompt_input : Any
    util_training_dir_output : Any

class DreamboothLoraTrainRequest(BaseModel):
    headless : Any
    print_only : Any
    pretrained_model_name_or_path : Any
    v2 : Any
    v_parameterization : Any
    logging_dir : Any
    train_data_dir : Any
    reg_data_dir : Any
    output_dir : Any
    max_resolution : Any
    learning_rate : Any
    lr_scheduler : Any
    lr_warmup : Any
    train_batch_size : Any
    epoch : Any
    save_every_n_epochs : Any
    mixed_precision : Any
    save_precision : Any
    seed : Any
    num_cpu_threads_per_process : Any
    cache_latents : Any
    cache_latents_to_disk : Any
    caption_extension : Any
    enable_bucket : Any
    gradient_checkpointing : Any
    full_fp16 : Any
    no_token_padding : Any
    stop_text_encoder_training_pct : Any
    # use_8bit_adam,
    xformers : Any
    save_model_as : Any
    shuffle_caption : Any
    save_state : Any
    resume : Any
    prior_loss_weight : Any
    text_encoder_lr : Any
    unet_lr : Any
    network_dim : Any
    lora_network_weights : Any
    dim_from_weights : Any
    color_aug : Any
    flip_aug : Any
    clip_skip : Any
    gradient_accumulation_steps : Any
    mem_eff_attn : Any
    output_name : Any
    model_list : Any  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length : Any
    max_train_epochs : Any
    max_data_loader_n_workers : Any
    network_alpha : Any
    training_comment : Any
    keep_tokens : Any
    lr_scheduler_num_cycles : Any
    lr_scheduler_power : Any
    persistent_data_loader_workers : Any
    bucket_no_upscale : Any
    random_crop : Any
    bucket_reso_steps : Any
    caption_dropout_every_n_epochs : Any
    caption_dropout_rate : Any
    optimizer : Any
    optimizer_args : Any
    noise_offset_type : Any
    noise_offset : Any
    adaptive_noise_scale : Any
    multires_noise_iterations : Any
    multires_noise_discount : Any
    LoRA_type : Any
    factor : Any
    use_cp : Any
    decompose_both : Any
    train_on_input : Any
    conv_dim : Any
    conv_alpha : Any
    sample_every_n_steps : Any
    sample_every_n_epochs : Any
    sample_sampler : Any
    sample_prompts : Any
    additional_parameters : Any
    vae_batch_size : Any
    min_snr_gamma : Any
    down_lr_weight : Any
    mid_lr_weight : Any
    up_lr_weight : Any
    block_lr_zero_threshold : Any
    block_dims : Any
    block_alphas : Any
    conv_dims : Any
    conv_alphas : Any
    weighted_captions : Any
    unit : Any
    save_every_n_steps : Any
    save_last_n_steps : Any
    save_last_n_steps_state : Any
    use_wandb : Any
    wandb_api_key : Any
    scale_v_pred_loss_like_noise_pred : Any
    scale_weight_norms : Any
    network_dropout : Any
    rank_dropout : Any
    module_dropout : Any

class DreamboothLoraTrainResponse(BaseModel):
    data: str = Field(title='data', description="dreamboot lora train response")

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock, prefix: str = None):
        if shared.cmd_opts.api_auth:
            self.credentials = dict()
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password
        
        self.app = app
        self.queue_lock = queue_lock
        self.prefix = prefix

        self.add_api_route(
            "tutorial",
            self.tutorial,
            methods=['POST'],
        )

        self.add_api_route(
            "dreambooth_lora_train",
            self.dreambooth_lora_train,
            methods=['POST'],
            response_model=DreamboothLoraTrainResponse,
        )

        self.add_api_route(
            "dreambooth_lora_folder_preparation",
            self.dreambooth_lora_folder_preparation,
            methods=['POST'],
        )


    def auth(self, creds: HTTPBasicCredentials = Depends(HTTPBasic())):
        if creds.username in self.credentials:
            if compare_digest(creds.password, self.credentials[creds.username]):
                return True
            
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Basic"
            })

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        if self.prefix:
            path = f'{self.prefix}/{path}'
        
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def tutorial(self, req: TutorialRequest):

        return TutorialResponse(
            data="tutorial output"
        )

    def dreambooth_lora_folder_preparation(self, req: DreamboothLoraFolderPreparationRequest):
        from library.dreambooth_folder_creation_gui import dreambooth_folder_preparation
        dreambooth_folder_preparation(
            util_output_model_name = req.util_output_model_name,
            util_training_images_dir_input = req.util_training_images_dir_input,
            util_training_images_repeat_input = req.util_training_images_repeat_input,
            util_instance_prompt_input = req.util_instance_prompt_input,
            util_regularization_images_dir_input = req.util_regularization_images_dir_input,
            util_regularization_images_repeat_input = req.util_regularization_images_repeat_input,
            util_class_prompt_input = req.util_class_prompt_input,
            util_training_dir_output = req.util_training_dir_output,
        )
        return {"dreambooth lora folder preparation finished"}
    
    def dreambooth_lora_train(self, req: DreamboothLoraTrainRequest):
        from lora_gui import train_model
        train_model(
            headless= req.headless,
            print_only= req.print_only,
            pretrained_model_name_or_path= req.pretrained_model_name_or_path,
            v2= req.v2,
            v_parameterization= req.v_parameterization,
            logging_dir= req.logging_dir,
            train_data_dir= req.train_data_dir,
            reg_data_dir= req.reg_data_dir,
            output_dir= req.output_dir,
            max_resolution= req.max_resolution,
            learning_rate= req.learning_rate,
            lr_scheduler= req.lr_scheduler,
            lr_warmup= req.lr_warmup,
            train_batch_size= req.train_batch_size,
            epoch= req.epoch,
            save_every_n_epochs= req.save_every_n_epochs,
            mixed_precision= req.mixed_precision,
            save_precision= req.save_precision,
            seed= req.seed,
            num_cpu_threads_per_process= req.num_cpu_threads_per_process,
            cache_latents= req.cache_latents,
            cache_latents_to_disk= req.cache_latents_to_disk,
            caption_extension= req.caption_extension,
            enable_bucket= req.enable_bucket,
            gradient_checkpointing= req.gradient_checkpointing,
            full_fp16= req.full_fp16,
            no_token_padding= req.no_token_padding,
            stop_text_encoder_training_pct= req.stop_text_encoder_training_pct,
            xformers= req.xformers,
            save_model_as= req.save_model_as,
            shuffle_caption= req.shuffle_caption,
            save_state= req.save_state,
            resume= req.resume,
            prior_loss_weight= req.prior_loss_weight,
            text_encoder_lr= req.text_encoder_lr,
            unet_lr= req.unet_lr,
            network_dim= req.network_dim,
            lora_network_weights= req.lora_network_weights,
            dim_from_weights= req.dim_from_weights,
            color_aug= req.color_aug,
            flip_aug= req.flip_aug,
            clip_skip= req.clip_skip,
            gradient_accumulation_steps= req.gradient_accumulation_steps,
            mem_eff_attn= req.mem_eff_attn,
            output_name= req.output_name,
            model_list=req.model_list,  
            max_token_length= req.max_token_length,
            max_train_epochs= req.max_train_epochs,
            max_data_loader_n_workers= req.max_data_loader_n_workers,
            network_alpha=req.network_alpha,
            training_comment= req.training_comment,
            keep_tokens= req.keep_tokens,
            lr_scheduler_num_cycles= req.lr_scheduler_num_cycles,
            lr_scheduler_power= req.lr_scheduler_power,
            persistent_data_loader_workers= req.persistent_data_loader_workers,
            bucket_no_upscale= req.bucket_no_upscale,
            random_crop= req.random_crop,
            bucket_reso_steps= req.bucket_reso_steps,
            caption_dropout_every_n_epochs= req.caption_dropout_every_n_epochs,
            caption_dropout_rate= req.caption_dropout_rate,
            optimizer= req.optimizer,
            optimizer_args= req.optimizer_args,
            noise_offset_type= req.noise_offset_type,
            noise_offset= req.noise_offset,
            adaptive_noise_scale= req.adaptive_noise_scale,
            multires_noise_iterations= req.multires_noise_iterations,
            multires_noise_discount= req.multires_noise_discount,
            LoRA_type= req.LoRA_type,
            factor= req.factor,
            use_cp= req.use_cp,
            decompose_both= req.decompose_both,
            train_on_input= req.train_on_input,
            conv_dim= req.conv_dim,
            conv_alpha= req.conv_alpha,
            sample_every_n_steps= req.sample_every_n_steps,
            sample_every_n_epochs= req.sample_every_n_epochs,
            sample_sampler= req.sample_sampler,
            sample_prompts= req.sample_prompts,
            additional_parameters= req.additional_parameters,
            vae_batch_size= req.vae_batch_size,
            min_snr_gamma= req.min_snr_gamma,
            down_lr_weight= req.down_lr_weight,
            mid_lr_weight= req.mid_lr_weight,
            up_lr_weight= req.up_lr_weight,
            block_lr_zero_threshold= req.block_lr_zero_threshold,
            block_dims= req.block_dims,
            block_alphas= req.block_alphas,
            conv_dims= req.conv_dims,
            conv_alphas= req.conv_alphas,
            weighted_captions= req.weighted_captions,
            unit= req.unit,
            save_every_n_steps= req.save_every_n_steps,
            save_last_n_steps= req.save_last_n_steps,
            save_last_n_steps_state= req.save_last_n_steps_state,
            use_wandb= req.use_wandb,
            wandb_api_key= req.wandb_api_key,
            scale_v_pred_loss_like_noise_pred= req.scale_v_pred_loss_like_noise_pred,
            scale_weight_norms=req.scale_weight_norms,
            network_dropout= req.network_dropout,
            rank_dropout= req.rank_dropout,
            module_dropout= req.module_dropout,


                
        )
        return DreamboothLoraTrainResponse(
            data="dreambooth lora train finished"
        )

def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/kohya/v1')