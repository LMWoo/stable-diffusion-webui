from PIL import Image, ImageFile

from modules import script_callbacks

# if you do not initialize the Image object
# Image.registered_extensions() returns only PNG
Image.init()

# PIL spits errors when loading a truncated image by default
# https://pillow.readthedocs.io/en/stable/reference/ImageFile.html#PIL.ImageFile.LOAD_TRUNCATED_IMAGES
ImageFile.LOAD_TRUNCATED_IMAGES = True


from extensions.kohya_ss.kohya_gui import UI
from extensions.kohya_ss.kohya_api import on_app_started

script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(UI)

print('Kohya GUI, API Initialized')
