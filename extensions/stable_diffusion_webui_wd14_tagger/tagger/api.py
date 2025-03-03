from typing import Callable
from threading import Lock
from secrets import compare_digest

from modules import shared
from modules.api.api import decode_base64_to_image
from modules.call_queue import queue_lock
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from tagger import utils
from tagger import api_models as models

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock, prefix: str = None) -> None:
        if shared.cmd_opts.api_auth:
            self.credentials = dict()
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.app = app
        self.queue_lock = queue_lock
        self.prefix = prefix

        self.add_api_route(
            'interrogate',
            self.endpoint_interrogate,
            methods=['POST'],
            response_model=models.TaggerInterrogateResponse
        )

        self.add_api_route(
            'interrogators',
            self.endpoint_interrogators,
            methods=['GET'],
            response_model=models.InterrogatorsResponse
        )

        self.add_api_route(
            'DataInterrogate',
            self.interrogate,
            methods=['POST']
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

    def endpoint_interrogate(self, req: models.TaggerInterrogateRequest):
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators.keys():
            raise HTTPException(404, 'Model not found')

        image = decode_base64_to_image(req.image)
        interrogator = utils.interrogators[req.model]

        with self.queue_lock:
            ratings, tags = interrogator.interrogate(image)

        return models.TaggerInterrogateResponse(
            caption={
                **ratings,
                **interrogator.postprocess_tags(
                    tags,
                    req.threshold
                )
            })

    def endpoint_interrogators(self):
        return models.InterrogatorsResponse(
            models=list(utils.interrogators.keys())
        )

    def interrogate(self, req: models.DataInterrogateRequest):
        from extensions.stable_diffusion_webui_wd14_tagger.tagger.ui import on_interrogate
        returnmsg = on_interrogate(
            output_modelname =req.output_modelname,
            batch_input_recursive=req.batch_input_recursive,
            batch_output_action_on_conflict=req.batch_output_action_on_conflict,
            batch_remove_duplicated_tag=req.batch_remove_duplicated_tag,
            batch_output_save_json=req.batch_output_save_json,
            interrogator=req.interrogator,
            threshold=req.threshold,
            additional_tags=req.additional_tags,
            exclude_tags=req.exclude_tags,
            sort_by_alphabetical_order=req.sort_by_alphabetical_order,
            add_confident_as_weight=req.add_confident_as_weight,
            replace_underscore=req.replace_underscore,
            replace_underscore_excludes=req.replace_underscore_excludes,
            escape_tag=req.escape_tag,
            unload_model_after_running=req.unload_model_after_running,
        )
        return {"Finshed Interrogate" : returnmsg}

def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/tagger/v1')
