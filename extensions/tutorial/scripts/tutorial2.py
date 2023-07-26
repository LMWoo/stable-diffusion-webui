
import os
import json
import gradio as gr

from modules import shared
from secrets import compare_digest
from typing import Callable
from threading import Lock
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from modules import script_callbacks
from modules.call_queue import queue_lock

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
        
        '''
        tutorial : API Path
        self.tutorial : API 함수
        '''
        self.add_api_route(
            'tutorial',
            self.tutorial,
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

    def tutorial(self):
        return {"Tutorial Response"}

def on_app_started(_, app: FastAPI):
    # tutorial Path : /tutorial/v1/tutorial
    Api(app, queue_lock, '/tutorial/v1')
    
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                submit = gr.Button(
                    value='Tutorial',
                )
    return [(interface, "Tutorial_TabName", "Tutorial_ID")]

'''
on_app_started, on_ui_tabs 호출해야 추가됨
'''
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)