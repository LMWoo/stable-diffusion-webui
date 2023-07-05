from PIL import Image, ImageFile

from modules import script_callbacks
# from tagger.api import on_app_started
# from tagger.ui import on_ui_tabs
from modules import script_callbacks


# if you do not initialize the Image object
# Image.registered_extensions() returns only PNG
Image.init()

# PIL spits errors when loading a truncated image by default
# https://pillow.readthedocs.io/en/stable/reference/ImageFile.html#PIL.ImageFile.LOAD_TRUNCATED_IMAGES
ImageFile.LOAD_TRUNCATED_IMAGES = True

print('Kohya Call')
# script_callbacks.on_app_started(on_app_started)

from extensions.kohya_ss.kohya_gui import UI
script_callbacks.on_ui_tabs(UI)
