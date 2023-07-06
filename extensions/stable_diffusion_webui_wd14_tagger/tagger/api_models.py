from typing import List, Dict

from modules.api import models as sd_models
from pydantic import BaseModel, Field


class TaggerInterrogateRequest(sd_models.InterrogateRequest):
    model: str = Field(
        title='Model',
        description='The interrogate model used.'
    )

    threshold: float = Field(
        default=0.35,
        title='Threshold',
        description='',
        ge=0,
        le=1
    )


class TaggerInterrogateResponse(BaseModel):
    caption: Dict[str, float] = Field(
        title='Caption',
        description='The generated caption for the image.'
    )


class InterrogatorsResponse(BaseModel):
    models: List[str] = Field(
        title='Models',
        description=''
    )


class DataInterrogateRequest(BaseModel):
    output_modelname: str
    # batch_input_glob: str,
    batch_input_recursive: bool
    # batch_output_dir: str,
    # batch_output_filename_format: str,
    batch_output_action_on_conflict: str
    batch_remove_duplicated_tag: bool
    batch_output_save_json: bool

    interrogator: str
    threshold: float
    additional_tags: str
    exclude_tags: str
    sort_by_alphabetical_order: bool
    add_confident_as_weight: bool
    replace_underscore: bool
    replace_underscore_excludes: str
    escape_tag: bool

    unload_model_after_running: bool