# LLM System Benchmarking tool [WIP]

LLM System benchmarking tool based on Promptflow. Grid-search your flow nodes to find the best performing combination of LLMs, prompts, search indexes etc. 

### Motivation 

LLM systems are increasingly composed of multiple components rather than a monolithic model. These system components need to be evaluated and optimized in order to develop robust applications that deliver great value. Because there are multiple options (e.g model provider, vector DB) to pick from for each component, it is challenging to systematically evaluate LLM systems and compare different combinations of components. Promptflow is a great tool to design and evaluate LLM systems but does not support grid-searching over flow nodes.
This tool extends Promptflow to help benchmark LLM systems by running a grid-search over a user-defined search space to find the ideal component combination for a give genAI task.

### Getting started 

#### 1. Install dependencies


####  Via the terminal, _from this repo's home directory_, run the following commands

```bash
conda env create --name pf_grid --file=./environment/conda.yaml
conda activate pf_grid
```

If you are not using VS Code you may need to run the following command to install the kernel

```bash
python -m ipykernel install --user --name pf_grid --display-name "pf_grid"
```
 **Create and populate a .env file in the home directory of this repository.** 
Use this [.sample.env file](.sample.env) as a guide. 


demo git

#### 2. Set-up connections to your LLM and Azure AI search
#### 3. Define your search space
#### 4. (Optional) modify the Promptflow template
#### 5. Run the grid-search script
#### 6. Evaluate results