import json
import os

from modules import ui_extra_networks, sd_hijack, shared


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self):
        from modules.textual_inversion.textual_inversion import Embedding
        import base64
        import requests
        from modules.api.models import EmbeddingsResponse

        url = "http://mwgpu.mydomain.blog:4000/sdapi/v1/embeddings"

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
        res = EmbeddingsResponse(**(response.json()))

        embeddings = {}
        for key in res.loaded.keys():
            newEmbedding = Embedding(None, key)
            newEmbedding.sd_checkpoint = res.loaded[key].sd_checkpoint
            newEmbedding.name = key
            newEmbedding.filename = key
            
            embeddings[key] = newEmbedding
            
        sd_hijack.model_hijack.embedding_db.word_embeddings.update(embeddings)
        # sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def list_items(self):
        for index, embedding in enumerate(sd_hijack.model_hijack.embedding_db.word_embeddings.values()):
            # path, ext = os.path.splitext(embedding.filename)
            yield {
                "name": embedding.name,
                "filename": embedding.name,
                "preview": None,
                "description": None,
                "search_term": None,
                "prompt": json.dumps(embedding.name),
                "local_preview": None,
                # "sort_keys": {'default': index, **self.get_sort_keys(embedding.filename)},

            }

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
