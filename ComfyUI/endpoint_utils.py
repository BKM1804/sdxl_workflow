import os
from s3_helper import S3Helper


def prepare_sdxl_checkpoint(checkpoint_path, job_id):
    if checkpoint_path.startswith("s3://"):
        s3_helper = S3Helper()
        bucket, key = s3_helper.s3_uri_to_bucket_key(checkpoint_path)
        s3_helper.download(bucket, key, f"{job_id}.zip")
    else:
        os.system(f"wget -O {job_id}.zip {checkpoint_path}")

    os.system(f"unzip {job_id}.zip -d {job_id}")
    os.system(f"rm {job_id}.zip")

    os.system(f"mv {job_id}/model/embeddings.pt models/embeddings/{job_id}.pt")
    os.system(f"mv {job_id}/model/lora.safetensors models/loras/sdxl_{job_id}.safetensors")
    img_name = os.listdir(f"{job_id}/model/data")[0]
    os.system(f"mv {job_id}/model/data/{img_name} input/{job_id}.png")

    os.system(f"rm -rf {job_id}")
    
    
def prepare_flux_checkpoint(checkpoint_path, job_id):
    if checkpoint_path.startswith("s3://"):
        s3_helper = S3Helper()
        bucket, key = s3_helper.s3_uri_to_bucket_key(checkpoint_path)
        s3_helper.download(bucket, key, f"models/loras/flux_{job_id}.safetensors")
    else:
        os.system(f"wget -O models/loras/flux_{job_id}.safetensors {checkpoint_path}")
