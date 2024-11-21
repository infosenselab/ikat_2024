# InfoSense @ iKAT 2024

This repo contains the code used for our TREC Interactive Knowledge Assistance Track (iKAT) submission titled "Passage Query Methods for Retrieval and Reranking in Conversational Agents".

## Setting up

- Create a conda environment and install `environment.yml` to it.

### Data

- Get the collection and index from the iKAT organizers.
- A helper script to combine the `ikat_2023_passage_index.tar.bz2.part*` files while getting an update after every file is added is available in `scripts/combine_index_helper.sh`.

- Create a `settings.py` file on the root with the following lines as needed:

```
COLLECTION_PATH = '/path_to/collection/'
INDEX_PATH = '/path_to/index/'
TOPICS_PATH = '/path_to/topics/'
BASELINES_PATH = '/path_to/baselines/'
QRELS_PATH = '/path_to/qrels/'
COLLECTION_HASHES_PATH = '/path_to/collection_hashes/'
MODELS_PATH = '/path_to/models'
```

### Evaluation

- For evaluation, we use `trec_eval`, available [here](https://github.com/usnistgov/trec_eval).
- Place it in the root directory, `cd trev_eval` and then run `make`. Run `make quicktest` to make sure it succesfully installed.
- Evaluations will run automatically during the run, but in you want to compare two files manually:
    - For a specific metric:`./trec_eval/trec_eval -m <metric> <qrels_file> <results_file>`. 
        - E.g. `./trec_eval/trec_eval -m ndcg_cut.5 /data/ikat_2024/qrels/2023-qrels.all-turns.txt ./output/2024-07-07-20-42-04_infosense_run.json`
    - For the full metrics:`./trec_eval/trec_eval -m all_trec <qrels_file> <results_file>`. 
        - E.g. `./trec_eval/trec_eval -m all_trec /data/ikat_2024/qrels/2023-qrels.all-turns.txt ./output/2024-07-07-20-42-04_infosense_run.json`
- NOTE: remember that qrels are not available for every single turn of the conversation.


### Validation

Using the 2024 folder:

- Validation is performed using the iKAT `run_validation` scripts available [here](https://github.com/irlabamsterdam/iKAT/tree/main/2023/scripts/run_validation).
- Place the `run_validation` in the root directory.
- Follow the installation steps on the README.md.
- In the `generate_run.py` file, change `run_file_name = PurePath(args.path_to_run_file).name` for `run_file_name = PurePath(args.path_to_run_file)`.
- You can see the log history in `run_validation/{run_name}.json.errlog`
- To run the validation: 
    1. Open TWO terminals. Active the same environment in BOTH and `cd` to the `./run_validation/` folder in BOTH, otherwise, the commands below won't work.
    1. One one of them, run `python ./passage_validator_servicer.py ./files/ikat_2023_passages_hashes.sqlite3`
        1. Wait until the `>> Service ready` message appears.
    1. On the other, run the main validation script with `python main.py ../output/<run_filename>.json -f /pathto/topics_folder/`.
        - For example, `python main.py ../output/2024-07-07-20-21-21/infosense_run.json -f /data/ikat_2024/topics/`.

## Performing a run

- Runs are performed from the `main_{run_name}.ipypy` files in the root.
- Adjust the `RUN SETTINGS` within the file as needed.

## Runs submitted

- `main_infosense_llama_pssgqrs_wghtdrerank.ipynb` was used to generate run 'infosense_llama_pssgqrs_wghtdrerank_1' and 'infosense_llama_pssgqrs_wghtdrerank_2'.
- `main_infosense_llama_short_long_qrs_1.ipynb` was used to generate run 'infosense_llama_short_long_qrs_2' (infosense_llama_short_long_qrs_1 was not submitted).
- `main_infosense_llama_short_long_qrs_2.ipynb` was used to generate run 'infosense_llama_short_long_qrs_3'.



