INFER_SCHEMA = {
    'sdxl_checkpoint_path': {
        'type': str,
        'required': True
    },
    'flux_checkpoint_path': {
        'type': str,
        'required': True
    },
    'prompt': {
        'type': str,
        'required': True
    },
    'width': {
        'type': int,
        'required': False,
        'default': 768
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1
    },
    'identifier': {
        'type': str,
        'required': False,
        'default': "sks"
    }
}
