import csv
import json
import logging
import requests
from pathlib import Path
import itertools
import datetime
#from unique_names_generator import get_random_name
import random
import string
from copy import deepcopy
import requests
import os



def create_cookiecutter(data, grid_step):
    data_dict = deepcopy(data)
    for node_id in data_dict:
        if node_id in ['llm','embedding','vector_store'] and node_id in grid_step.keys():
            data_dict[node_id]=grid_step[node_id]
        elif node_id in ['llm','embedding','vector_store'] and node_id not in grid_step.keys():
            del data_dict[node_id]
    return data_dict


def generate_run_id():
    #name = get_random_name(separator="_", style="lowercase")
    timestamp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return 'run_'+ timestamp_id


def flatten_dict(variant_dict):
    variant_name = list(variant_dict.keys())[0]
    variant_values = variant_dict[variant_name]
    result_dict = {'variant_name': variant_name}
    result_dict.update(variant_values)
    return result_dict

def cartesian_product(*lists):
    cartesian_product = []
    for items in itertools.product(*lists):
        merged_dict = {}
        for item in items:
            merged_dict.update(item)
        cartesian_product.append(merged_dict)
    return cartesian_product

def duplicate_dict(dictionary):
    result = [dictionary]
    for key, value in dictionary.items():
        if isinstance(value, list):
            temp_result = []
            for item in value:
                for res in result:
                    temp_dict = res.copy()
                    temp_dict[key] = item
                    temp_result.append(temp_dict)
            result = temp_result
    return result

def download_artifacts(run_id, datastore, asset_name, ws, output_path):
    region = ws.location
    workspace_name = ws.name
    subscription_id = ws.subscription_id
    resource_group = ws.resource_group

    if region == "centraluseuap":
        url = f"https://int.api.azureml-test.ms/history/v1.0/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/rundata"
    else:
        url = f"https://ml.azure.com/api/{region}/history/v1.0/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/rundata"
        payload = {
            "runId": run_id,
            "selectRunMetadata": True
        }
    response = requests.post(url, json=payload, headers=ws._auth.get_authentication_header())
    if response.status_code != 200:
        raise Exception(f"Failed to get output asset id for run {run_id} because RunHistory API returned status code {response.status_code}. Response: {response.text}")
    output_asset_id = response.json()["runMetadata"]["outputs"][asset_name]["assetId"]
    if region == "centraluseuap":
        url = f"https://int.api.azureml-test.ms/data/v1.0/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/dataversion/getByAssetId"
    else:
        url = f"https://ml.azure.com/api/{region}/data/v1.0/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/dataversion/getByAssetId"
    payload = {
            "value": output_asset_id,
        }
    response = requests.post(url, json=payload, headers=ws._auth.get_authentication_header())
    if response.status_code != 200:
        raise Exception(f"Failed to get asset path for asset id {output_asset_id} because Data API returned status code {response.status_code}. Response: {response.text}")
    data_uri = response.json()["dataVersion"]["dataUri"]
    relative_path = data_uri.split("/paths/")[-1]

    destination_path = os.path.join(output_path,run_id)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    datastore.download(destination_path, prefix=relative_path +'flow_artifacts', overwrite=True)