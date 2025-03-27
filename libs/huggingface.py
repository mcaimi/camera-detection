#!/usr/bin/env python

try:
    import os
    import huggingface_hub as hf
    from .parameters import Parameters
except Exception as e:
    raise e


# download a checkpoint from HuggingFace
def pullFromHuggingfaceHub(pullConfig: Parameters) -> dict:
    # Download model checkpoint from HuggingFace repositories
    os.environ["HF_HOME"]: str = pullConfig.huggingface.hfHomePath
    remote_model_objects: dict = {}
    mistral_models_path: str = "/".join((pullConfig.huggingface.modelsPath, pullConfig.huggingface.modelName))
    os.makedirs(mistral_models_path, exist_ok=True)

    print(f"Downloading model checkpoint: {pullConfig.huggingface.modelName}")
    for name in pullConfig.huggingface.filenames:
        model_path = hf.snapshot_download(repo_id=pullConfig.huggingface.modelName,
                                          allow_patterns=pullConfig.huggingface.filenames,
                                          revision="main",
                                          token=pullConfig.huggingface.apiToken,
                                          local_dir=mistral_models_path)
        print(f"Downloaded model checkpoint {model_path}")
        for n in pullConfig.huggingface.filenames:
            remote_model_objects["/".join((model_path, n))] = n

    # return pulled file hash
    return remote_model_objects
