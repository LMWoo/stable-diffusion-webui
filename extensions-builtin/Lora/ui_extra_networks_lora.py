import json
import os
import lora

from modules import shared, ui_extra_networks


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        def lora_refresh():
            import requests
            import json
            import base64

            url = 'http://mwgpu.mydomain.blog:4000/sdapi/v1/refresh-loras'

            auth = 'user:password'
            auth_bytes = auth.encode('UTF-8')

            auth_encoded = base64.b64encode(auth_bytes)
            auth_encoded = bytes(auth_encoded)
            auth_encoded_str = auth_encoded.decode('UTF-8')

            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Basic ' + auth_encoded_str
            }

            response = requests.request('POST', url=url, headers=headers)
            

        lora_refresh()
        
        lora.available_loras.clear()
        lora.available_lora_aliases.clear()
        lora.forbidden_lora_aliases.clear()
        lora.available_lora_hash_lookup.clear()
        lora.forbidden_lora_aliases.update({"none": 1, "Addams": 1})

        import requests
        import json
        import base64

        url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/loras"

        auth = 'user:password'
        auth_bytes = auth.encode('UTF-8')

        auth_encoded = base64.b64encode(auth_bytes)
        auth_encoded = bytes(auth_encoded)
        auth_encoded_str = auth_encoded.decode('UTF-8')

        headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + auth_encoded_str
        }

        response = requests.request("GET", url=url, headers=headers)
        response = response.json()
        for loraData in response:
            name = os.path.splitext(os.path.basename(loraData["path"]))[0]
            filename = loraData["path"]
            metadata = loraData["metadata"]

            entry = lora.LoraOnDisk(name, filename, metadata)
            
            lora.available_loras[name] = entry

            # lora.forbidden_lora_aliases[entry.alias.lower()] = 1

            # lora.available_lora_aliases[name] = entry
            # lora.available_lora_aliases[entry.alias] = entry
        # lora.list_available_loras()

    def list_items(self):
        for index, (name, lora_on_disk) in enumerate(lora.available_loras.items()):
            path, ext = os.path.splitext(lora_on_disk.filename)

            alias = lora_on_disk.get_alias()

            yield {
                "name": name,
                "filename": path,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(lora_on_disk.filename),
                "prompt": json.dumps(f"<lora:{alias}:") + " + opts.extra_networks_default_multiplier + " + json.dumps(">"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(lora_on_disk.metadata, indent=4) if lora_on_disk.metadata else None,
                # "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},

            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]

