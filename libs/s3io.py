#!/usr/bin/env python

try:
    import boto3
    import os
    import sys
    from .parameters import Parameters
    import threading
except Exception as e:
    raise e

# Gigabyte Unit in bytes
GB = 1024 ** 3


# shamelessly stolen from aws docs :D
class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


# S3 I/O class
class S3IO(object):
    def __init__(self, config: Parameters):
        self.config: Parameters = config
        self.s3api = None

    def s3Open(self):
        # download model from s3 if needed
        try:
            # connect to MinIO and prepare buckets
            print(f"Accessing S3 endpoint {self.config.params.url} with ACCESS_KEY {self.config.params.accessKey}...")

            # instantiate connection
            self.s3api = boto3.client("s3",
                                      endpoint_url=self.config.params.url,
                                      aws_access_key_id=self.config.params.accessKey,
                                      aws_secret_access_key=self.config.params.secretKey)
        except Exception as e:
            print(f"Caught exception: {e}")

    # create required buckets on a target S3-compatible endpoint
    def prepareBuckets(self):
        # Create the models bucket
        available_buckets = [buckets["Name"] for buckets in self.s3api.list_buckets()["Buckets"]]
        for bckname in self.config.s3.bucket_list:
            print(f"-> Creating bucket {bckname}...")
            if bckname not in available_buckets:
                try:
                    self.s3api.create_bucket(Bucket=bckname)
                except Exception as e:
                    print(f"Failure during bucket creation due to this error: {e}")
            else:
                print(f"--> Bucket ({bckname}) Already Exists. Skipping...")

    # try to pull the model checkpoint binary file from an S3 bucket
    def pullCheckpointFromS3(self):
        # create folder to store training data
        models_path = "/".join((self.config.huggingface.modelsPath,
                                self.config.huggingface.modelName))
        os.makedirs(models_path, exist_ok=True)

        # get list of data files
        try:
            for file in self.config.huggingface.filenames:
                if not os.path.exists("/".join((models_path, file))):
                    print(f"Downloading file: {file} to {models_path}")
                    self.s3api.download_file(self.config.huggingface.modelBucket,
                                             file,
                                             "/".join((models_path, file)))
                else:
                    print(f"File {file} already downloaded.")
        except Exception as e:
            print(f"Caught Exception {e}")

    # check if a specific object exists in a remote bucket
    # checks whether a file exists in a remote bucket
    def check_exists(self, bucket, filename):
        rsp = self.s3api.list_objects_v2(Bucket=bucket, Prefix=filename)
        try:
            contents = rsp.get("Contents")
            files = [obj.get("Key") for obj in contents]
            if filename in files:
                return True
            else:
                return False
        except Exception as e:
            raise e

    # upload data to an s3 bucket
    def pushCheckpointToS3(self, remote_objects: dict):
        transfer_config = boto3.TransferConfig(multipart_threshold=self.config.s3.multipart_threshold_gb * GB,
                                               use_threads=self.config.s3.use_thread)

        try:
            for k in remote_objects.keys():
                if not self.check_exists(self.s3api, self.config.huggingface.modelBucket, remote_objects[k]):
                    print(f"Uploading {remote_objects[k]} to MinIO bucket {self.config.huggingface.modelBucket}")
                    self.s3api.upload_file(k, self.config.huggingface.modelBucket,
                                           remote_objects[k],
                                           Callback=ProgressPercentage(k),
                                           Config=transfer_config)
                    print("---")
                else:
                    print(f"File {k} already exists in {self.config.huggingface.modelBucket}")
        except boto3.ClientError as e:
            print(f"S3 Exception: {e.response['Error']['Code']}, trace: {e}")
        except Exception as e:
            print(f"Caught exception: {e}")

        print("Upload Complete.")
