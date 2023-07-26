import json
import os

from modules import shared, ui_extra_networks


class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Hypernetworks')

    # hypernetworks 모델 Refresh함수
    def refresh(self):
        from modules.hypernetworks.hypernetwork import Hypernetwork
        import base64
        import requests
        from modules import sd_hijack
        from modules.api.models import HypernetworkItem

        url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/hypernetworks"

        auth = 'user:password'
        auth_bytes = auth.encode('UTF-8')
        
        auth_encoded = base64.b64encode(auth_bytes)
        auth_encoded = bytes(auth_encoded)
        auth_encoded_str = auth_encoded.decode('UTF-8')

        headers = {
            'accept': 'application/json',
            'Authorization': 'Basic ' + auth_encoded_str,
        }

        response = requests.get(url=url, headers=headers)
        print(response.json())
        items = response.json()
        shared.hypernetworks.clear()
        
        for item in items:
            newItem = Hypernetwork(name = item["name"])
            shared.hypernetworks[item["name"]] = item["name"]
        
    def refresh_req(self):
        shared.reload_hypernetworks()

    def list_items(self):
        for index, (name, path) in enumerate(shared.hypernetworks.items()):
            path, ext = os.path.splitext(path)

            yield {
                "name": name,
                "filename": path,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(path),
                "prompt": json.dumps(f"<hypernet:{name}:") + " + opts.extra_networks_default_multiplier + " + json.dumps(">"),
                "local_preview": f"{path}.preview.{shared.opts.samples_format}",
                # "sort_keys": {'default': index, **self.get_sort_keys(path + ext)},

            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.hypernetwork_dir]

