import yaml
import itertools
from copy import deepcopy
import argparse
from flowgrid import utils
import logging
import os
import json
import subprocess
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azure.ai.ml import MLClient
from promptflow.azure import PFClient
from promptflow.entities import Run
import datetime 
import threading
import time
from dotenv import load_dotenv
from tabulate import tabulate



# Get a logger which allows us to log events that occur when running the program.
logger = logging.getLogger("gridsearch")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument('--sweep', type=str, default='sweep_definition.yaml')
parser.add_argument('--evaluation_data', type=str, default='./evaluation_data/data.jsonl', help='Path to the input data file. for the batch evaluation')
parser.add_argument('--pf_template', type=str, default='./flows/flow_template')
parser.add_argument('--simulate', type=bool, default=False, help='If True, the generated flow runs will not be submitted to AzureML.')


args = parser.parse_args()

logger.debug(f"Loading environment variables")
load_dotenv('.env')

subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")
pf_ci = os.environ.get("PF_CI")

logger.debug(f"Azure Subscription ID: {subscription_id}")


sweep_run_id = utils.generate_run_id()
logger.info(f"Generated Sweep Run ID: {sweep_run_id}")

with open(args.sweep, 'r') as file:
    sweep_config = yaml.safe_load(file)


search_space =[]
for node in sweep_config.get('search_space'):
    if node in ['llm','embedding','vector_store']:
        grid = []
        node_values = sweep_config.get('search_space').get(node)
        for variant in node_values:
            step_dict ={}
            step_dict[node] = utils.flatten_dict(variant)
            grid.append(step_dict)
        grid_final = []
        for step in grid:
            
            node_id = list(step.keys())[0]    
            fixed = {key:value for key,value in step.get(node_id).items() if not isinstance(value, dict)}
            for key, value in step.get(node_id).items():
                option_list =[]
                if isinstance(value, dict):
                    options = utils.duplicate_dict(value)
                    for option in options:
                        f = deepcopy(fixed)
                        f[key] = option.get('values')
                        option_list.append(f)    
            
            if len(option_list)>1:
                for opt in option_list:
                    grid_final.append({node_id: opt})
            else:
                grid_final.append({node_id: fixed})
        search_space.append(grid_final)

grid_search_steps = utils.cartesian_product(*search_space)

logger.info(f"Number of Grid Search Steps: {len(grid_search_steps)}")


file_path = "./flows/flow_template/cookiecutter_template.json"
cookiecutter_file_path = "./flows/flow_template/cookiecutter.json"


with open(file_path, "r") as file:
    data = json.load(file)

for i, search_step_dict in enumerate(grid_search_steps):
    directory = os.path.join('./flows/flow_versions',sweep_run_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    step_dict = utils.create_cookiecutter(data, search_step_dict)
    step_dict["flow_name"] = f"grid_step_{i}"
    
    with open(cookiecutter_file_path, "w") as file:
        json.dump(step_dict, file, indent=4)

    logger.debug(f"Generating PromptFlow flow for Grid Step: {i}")
    subprocess.run([f"cd {directory} && cookiecutter ../../flow_template --no-input --skip-if-file-exists"], shell=True)
    logger.debug(f"Flow Generated for Grid Step: {i} at {directory}/rag_flow_grid_step_{i}")

logger.info(f"Created {len(grid_search_steps)} PromptFlow flows in {directory} for the sweep run: {sweep_run_id}")

if args.simulate:
    logger.info(f"Simulate is set to True, not submitting the generated flows to AzureML.")
else:

    logger.info(f"Connecting to the AzureML workspace.")
    credential = DefaultAzureCredential()
  
    # Get a handle to workspace, it will use config.json in current and parent directory.
    pf = PFClient.from_config(credential=credential)

    
    def create_and_run_flow(i):
    # Apply your function to the dictionary here
        flow = os.path.join("./flows/flow_versions", f"{sweep_run_id}/rag_flow_grid_step_{i}")
        data ="./evaluation_data/data.jsonl"
        # create run
        
        base_run = pf.run(
            flow=flow,
            data=data,
            runtime=pf_ci,
            column_mapping={
            "question": "${data.question}"
            }
        )
        logger.info(f"Created run {base_run.name}")
        return base_run


    # Create a thread for each dictionary
    runs = []
    def run_flow_and_wait(i):
        logger.info(f"Submitting run for Grid Step: {i}")
        run = create_and_run_flow(i)
        completed = False
        while not completed:
            status = pf.runs.get(run.name).status
            if status in ["Completed", "Failed"]:
                completed = True
            else:
                logger.debug(f"Run {run.name} is in status {status}")
                time.sleep(30)
        runs.append(run)
        logger.info(f"Run {run.name} completed with status {status}")

    eval_runs = []
    def create_and_run_eval_flow(i):
    # Apply your function to the dictionary here
        flow = "./flows/evaluation_flow"
        data ="./evaluation_data/data.jsonl"
        # create run
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = "eval_"+runs[i].name
        eval_run = Run(
                flow='./flows/evaluation_flow',
                data='./evaluation_data/data.jsonl',
                run=runs[i].name,
                column_mapping={
                    "answer": "${run.outputs.response}",
                    "ground_truth": "${data.ground_truth}"         },
                # un-comment the resources parameter assignment
                # and update the size of the compute and also
                # comment the runtime parameter assignment to
                # enable automatic runtime.
                # Reference: COMPUTE_RUNTIME
                # resources={"instance_type": "Standard_E4ds_v4"},
                name=f"{experiment_name}_eval_{timestamp}",
                display_name=f"{experiment_name}_eval_{timestamp}",
            )
        eval_job = pf.runs.create_or_update(eval_run)
        logger.info(f"Created run {eval_job.name}")
        eval_runs.append({"base_run":runs[i].name, "eval":eval_job.name})
        return eval_job

    # Create a thread for each dictionary
    threads = []
    for i, search_step_dict in enumerate(grid_search_steps):
        thread = threading.Thread(target=run_flow_and_wait, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    run_statuses = [pf.runs.get(run.name).status for run in runs]    
    success_count = run_statuses.count("Completed")

    logger.info(f"Grid Search Run Completed. {success_count} out of {len(runs)} runs completed successfully.")
    logger.info(f"Dowloading the artifacts for the completed runs.")
    logger.info(f"Connecting to AML workspace.")
    
    interactive_auth = InteractiveLoginAuthentication()
    aml_ws = Workspace.get(workspace_name, subscription_id=subscription_id, resource_group=resource_group, auth=interactive_auth)
    region = aml_ws.location
    datastore = aml_ws.get_default_datastore()

    
    for run in runs:
        run_id = run.name
        utils.download_artifacts(run_id, 
                                 datastore, 
                                 "debug_info", 
                                 aml_ws,
                                 f"./run_outputs/{sweep_run_id}/run_{run_id}")
    
    logger.info(f"Downloaded the artifacts for the completed runs to ./run_outputs/{sweep_run_id}")

    def run_eval_flow_and_wait(i):
        logger.info(f"Submitting evaluation run for Grid Step: {i}")
        eval_run = create_and_run_eval_flow(i)
        completed = False
        while not completed:
            status = pf.runs.get(eval_run.name).status
            if status in ["Completed", "Failed"]:
                completed = True
            else:
                logger.debug(f"Run {eval_run.name} is in status {status}")
                time.sleep(30)
        logger.info(f"Run {eval_run.name} completed with status {status}")

    run_eval = input("Batch runs completed, do you want to run evaluations? Y/n ")
    if run_eval.lower() in ["y","n"]:
        if run_eval.lower() == "y":
            for i, run in enumerate(runs):
                thread = threading.Thread(target=run_eval_flow_and_wait, args=(i,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            logger.info(f"Created and submitted evaluation runs for all the completed runs.")

            eval_run_statuses = [pf.runs.get(eval_run_id.get('eval')).status for eval_run_id in eval_runs]
            eval_success_count = eval_run_statuses.count("Completed")
            logger.info(f"Evaluation runs completed. {eval_success_count} out of {len(eval_runs)} runs completed successfully.")
            for eval_run in eval_runs:
                logger.info(f" Evaluation results for run {eval_run.get('base_run')} : {pf.get_metrics(eval_run.get('eval'))}")
        else:
            logger.info(f"Exiting the program.")

    else:
        raise ValueError("Invalid input. Please enter Y or n")

