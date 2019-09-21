# %% markdown
# Importing video data from ibm COS

import base64
# %%
import json
import os
import pickle
import random
import sys
import time

import cv2
import ibm_boto3
import numpy as np
import pandas as pd
from ibm_botocore.client import Config
from tqdm import tqdm

# %%
cred_path = "/home/alex/Documents/MIDS/w210/FinalProject/w210-credentials.json"
# cred_path = "/tmp/w210-credentials.json"
with open(cred_path, "r") as f:
    creds = json.load(f)

auth_endpoint = 'https://iam.bluemix.net/oidc/token'
service_endpoint = 'https://s3.us.cloud-object-storage.appdomain.cloud'

# Store relevant details for interacting with IBM COS store and uploading data
cos = ibm_boto3.resource('s3',
                         ibm_api_key_id=creds['apikey'],
                         ibm_service_instance_id=creds['resource_instance_id'],
                         ibm_auth_endpoint=auth_endpoint,
                         config=Config(signature_version='oauth'),
                         endpoint_url=service_endpoint)
# %%
# FIXME: need to figure out how to install aspera for massive download speed
# boost. Surprisingly, the docs suck
#
# from ibm_s3transfer.aspera.manager import AsperaTransferManager
# cos = ibm_boto3.client('s3',
#                          ibm_api_key_id=creds['apikey'],
#                          ibm_service_instance_id=creds['resource_instance_id'],
#                          ibm_auth_endpoint=auth_endpoint,
#                          config=Config(signature_version='oauth'),
#                          endpoint_url=service_endpoint)
# transfer_manager = AsperaTransferManager(cos)
# %%
bucket = cos.Bucket('w210-finalproject')
# %%
files = list(bucket.objects.all())
# %%
# savepath = "/home/alex/Documents/MIDS/w210/FinalProject/tmp"
savepath = '/tmp/vids'
os.makedirs(savepath, exist_ok=True)
# %%
for file in tqdm(files):
    ext = file.key.split('.')[-1]
    if 'mp4' not in ext and 'avi' not in ext:
        continue
    filename = os.path.join(savepath, file.key.split('/')[-1])
    bucket.download_file(file.key, filename)

# %%
vids = os.listdir(savepath)
cap = cv2.VideoCapture(os.path.join(savepath, vids[0]))
cap.isOpened()
# %%
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('vid', gray)

# %%
