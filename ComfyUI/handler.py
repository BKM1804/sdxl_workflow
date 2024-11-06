''' infer.py for runpod worker '''

import os
import time
import runpod
from runpod.serverless.utils.rp_validator import validate

from schema import INFER_SCHEMA

from workflow import init_foundaton_nodes, init_specific_nodes, main
from s3_helper import S3Helper
from endpoint_utils import prepare_sdxl_checkpoint, prepare_flux_checkpoint

AWS_S3_BUCKET_NAME = "lustylens"
AWS_S3_IMAGES_PATH = "generations"

s3_helper = S3Helper()
since = time.time()
init_foundaton_nodes()
print(f"Time taken to init nodes: {time.time() - since} seconds")


def run(job):
    '''
    Run training or inference job.
    '''
    job_input = job['input']
    print(f"Job input: {job_input}")
    job_id = job['id']
        
    validate_train = validate(job_input, INFER_SCHEMA)
    print(f"Validated input: {validate_train}")
    if 'errors' in validate_train:
        return {'error': validate_train['errors']}
    
    validate_train = validate_train['validated_input']

    since = time.time()
    prepare_sdxl_checkpoint(validate_train['sdxl_checkpoint_path'], job_id)
    prepare_flux_checkpoint(validate_train['flux_checkpoint_path'], job_id)
    init_specific_nodes(job_id=job_id)
    output = []
    for _ in range(validate_train['num_images']):
        res = main(
            job_id=job_id,
            prompt=validate_train['prompt'],
            width=validate_train['width'],
            height=validate_train['height'],
        )
        output.append(os.path.join("output", res["ui"]["images"][0]["subfolder"], res["ui"]["images"][0]["filename"]))
    print(f"Time taken: {time.time() - since} seconds")
    
    # upload images to s3
    res = []
    for image in output:
        uri = s3_helper.upload(image, AWS_S3_BUCKET_NAME, os.path.join(AWS_S3_IMAGES_PATH, os.path.basename(image)))
        res.append(s3_helper.s3_uri_to_link(uri))

    return {
        "images": res
    }


runpod.serverless.start({"handler": run})