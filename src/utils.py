import numpy as np
from collections import defaultdict
import subprocess
import json
import os
import torch

from pyserini.search import LuceneSearcher
from sentence_transformers import util

import settings


def use_llama(model, tokenizer, system_message_content: str, user_message_content: str):
    
    # define the messages for the model to process
    messages = [
        {"role": "system", "content": f"{system_message_content}"},
        {"role": "user", "content": f"{user_message_content}"},
    ]


    # apply template to the messages and convert to input tensors
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   
        return_tensors="pt" 
    ).to(model.device) # move tensors to the model's device

    # define termination tokens
    terminators = [
        tokenizer.eos_token_id, 
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    
    # generate response using the model
    outputs = model.generate( 
        input_ids,
        max_new_tokens=256*2,
        eos_token_id=terminators,
        do_sample=False, # no randomness
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id, # avoid `Setting pad_token_id to eos_token_id:128009 for open-end generation` message
    )
    
    # extract and decode the generated response
    response_encoded = outputs[0][input_ids.shape[-1]:]
    response_decoded= tokenizer.decode(response_encoded, skip_special_tokens=True)
    
    return response_decoded



def prepare_output_for_json(turn_id, sorted_responses, sorted_responses_provenance, reranked_passages, searcher, relevant_ptkbs):

    # limit reranked_passages to 1000 entries
    reranked_passages = reranked_passages[:1000]

    # initialize a dictionary to store data for the current turn
    turn_data = {
        "turn_id": turn_id, 
        "responses": []    
    }

    # iterate over the list of sorted responses
    for i in range(len(sorted_responses)):
        response_data = {
            "rank": i + 1,  
            "text": sorted_responses[i],  
            "ptkb_provenance":  [int(ptkb['id']) for ptkb in relevant_ptkbs],
            "passage_provenance": [      
                {
                    "id": hit.docid,  
                    "text": json.loads(searcher.doc(hit.docid).raw())['contents'],  
                    "score": hit.score, 
                    "used": any(hit.docid == prov.docid for prov in sorted_responses_provenance[i]) # flag to indicate the passage was used to generate the response
                } for hit in [reranked_passages][i]  # iterate over the hits for the current response
            ]
            
        }
        
        turn_data["responses"].append(response_data)
    
    return turn_data


def document_searcher(
        query_id: str,                          
        query: str,                             
        k: int,                                 
        searcher: LuceneSearcher,               
        ):
    
    # perform the search with the given query and retrieve the top k documents
    hits = searcher.search(q=query, k=k)
    
    # check if any documents were retrieved
    if len(hits) == 0:
        raise Exception(f"No hits for QueryID: {query_id}\nQuery: {query}")

    return hits



def join_and_cap_passages(top_passages, max_words=300):
    capped_passages = []

    for i, passage in enumerate(top_passages, 1):
        words = passage.split()
        if len(words) > max_words:
            # truncate the passage to the maximum word count
            passage = " ".join(words[:max_words])
        capped_passages.append(f"Passage {i}: {passage}")

    # join all passages with double newlines
    text = "\n\n".join(capped_passages)
    return text



def get_true_ptkbs(ptkb_qrels, conversation_id):

    true_ids = []    
    id_found = False
    
    # iterate
    for line in ptkb_qrels:
        
        # split
        parts = line.strip().split()
        
        # extract elements
        id = parts[0]
        item_id = parts[2]
        boolean = parts[3]
        
        # check if the conversation ID matches
        if id == conversation_id:
            id_found = True
            
            # check if the boolean value is '1'
            if boolean == '1':
                true_ids.append(item_id)
    
    if not id_found:
        true_ids.append('no ptkb qrels for turn')
    
    return true_ids


def has_qrels_for_conversation(passage_qrels, conversation_prefix):
    # convert passage_qrels into numpy array
    data = np.array([line.strip().split(' ') for line in passage_qrels])
    
    # ensure string
    conversation_prefix = str(conversation_prefix)
    
    # filter rows
    filtered_rows = [row for row in data if row[0].startswith(conversation_prefix) and 
                     (len(row[0]) == len(conversation_prefix) or not row[0][len(conversation_prefix)].isdigit())]
    
    
    
    return len(filtered_rows) > 0



def parse_metrics(text):
    metrics = {}
    lines = text.strip().split('\n')
    for line in lines:
        if line:
            parts = line.split()
            metric_name = parts[0]
            value = float(parts[-1])
            metrics[metric_name] = value
    return metrics


def combine_performance_metrics(pre_str, post_str):
    pre_metrics = parse_metrics(pre_str)
    post_metrics = parse_metrics(post_str)
    
    combined_metrics = "{:<20} {:<15} {:<15}\n".format("Metric", "Pre-Reranking", "Post-Reranking")
    combined_metrics += "-" * 50 + "\n"
    
    for metric in pre_metrics:
        pre_value = pre_metrics[metric]
        post_value = post_metrics.get(metric, "N/A")
        combined_metrics += "{:<20} {:<15} {:<15}\n".format(metric, pre_value, post_value)
    
    return combined_metrics


def combine_hits(query_list, query_id, num_docs, searcher, verbose_output_filename):
    
    combined_hits = []
    seen_docs = set()
    query_source_counts = defaultdict(int)
    query_source_docs = defaultdict(list)
    
    for i, query in enumerate(query_list):
        
        # save all the prints to a text file for tracking system progress
        with open(verbose_output_filename, 'a') as f:
            f.write('\n## Searching for documents....\n')
            f.write(f'\n{query}\n')
        
        
        # get the top k documents for each query
        interim_hits = document_searcher(
            query_id=query_id,
            query=query,
            k=num_docs,
            searcher=searcher
        )
                        
        # add only unique hits to the combined list
        for hit in interim_hits:
            if hit.docid not in seen_docs:
                combined_hits.append(hit)
                seen_docs.add(hit.docid)
                query_number = i + 1  
                query_source_counts[query_number] += 1
                query_source_docs[query_number].append(hit.docid)
    
    return combined_hits, query_source_counts, query_source_docs


def full_evaluations_join(ndcg_cut_3_eval, result_evaluations):
    
    # put ndcg3 in the right place
    # split the evaluation string into a list of lines
    lines = result_evaluations.strip().split("\n")

    # find the index of the line containing "ndcg"
    ndcg_index = next(i for i, line in enumerate(lines) if line.startswith("ndcg"))

    # insert the ndcg_cut_3 string after the "ndcg" line
    lines.insert(ndcg_index + 1, ndcg_cut_3_eval.strip())

    
    return "\n".join(lines)



def run_generate_run(path_to_run_file, generate_run_file_path='./run_validation/generate_run.py'):
    result = subprocess.run(
        ['python3', generate_run_file_path, path_to_run_file],
        capture_output=True,
        text=True
    )
    
    # check if successful
    if result.returncode != 0:
        print("Error executing generate_run.py.")
        print(result.stderr) 



def measure_intrarun_ranking_performance(turn_json, run_name, output_filename, passage_qrels_filename, section_type='turn'):
            
    # Measure the results of the current turn
    turn_result_json = {
        "run_name": run_name,
        "run_type": 'automatic',
        #"eval_response": True,
        "turns": [turn_json] if section_type == "turn" else turn_json if section_type == "conversation" else None

    }
    
    # write temp file
    turn_result_json_tmpfile = output_filename.replace('.json', '_temp.json')
    with open(turn_result_json_tmpfile, 'w') as f:
        if isinstance(turn_result_json, list):
            for line in turn_result_json:
                f.write("%s\n" % line)
        elif isinstance(turn_result_json, dict):
            json.dump(turn_result_json, f, indent=4)
    

    # convert to TRECRun format
    run_generate_run(turn_result_json_tmpfile)
    
    
    # full results    
    turn_result_evaluations_ndcg_cut_3 = run_trec_eval(passage_qrels_filename, f"{turn_result_json_tmpfile.replace('.json', '.json.run')}", metric_list=['ndcg_cut.3'])
    turn_result_evaluations = run_trec_eval(passage_qrels_filename, f"{turn_result_json_tmpfile.replace('.json', '.json.run')}", metric_list=['ndcg', 'ndcg_cut', 'P', 'recall', 'map'])

    turn_result_evaluations = full_evaluations_join(turn_result_evaluations_ndcg_cut_3, turn_result_evaluations)
    
    
    # remove the temporary files
    if os.path.exists(turn_result_json_tmpfile):
        os.remove(turn_result_json_tmpfile)
        os.remove(turn_result_json_tmpfile.replace('.json', '.json.run'))


    return turn_result_evaluations


def check_quick_retrieval_success(preliminary_hits, expected_qrel_passages, query_source_counts, query_source_docs, verbose_output_filename):
    retrieved_document_ids = [hit.docid for hit in preliminary_hits]
    matched_ids = [passage for passage in expected_qrel_passages if passage in retrieved_document_ids]
    not_matched_ids = [passage for passage in expected_qrel_passages if passage not in retrieved_document_ids]
    count = len(matched_ids)
    total = len(expected_qrel_passages)
    fraction_found = count / total

    # save all the prints to a text file for tracking system progress
    with open(verbose_output_filename, 'a') as f:
        f.write('')

        f.write(f'\n{count} items in the expected_qrel_passages set are in retrieved_document_ids set.\n')
        f.write(f'\nFraction found: {count}/{total} = {fraction_found:,}\n\n')
        f.write(f'You did not find:\n{not_matched_ids}\n\n')

        # display the count of documents found from each query
        for query_number, doc_count in query_source_counts.items():
            matched_doc_count = len([doc for doc in query_source_docs[query_number] if doc in matched_ids])
            matched_docs_from_query = [doc for doc in query_source_docs[query_number] if doc in matched_ids]
            f.write(f'\n-> Query {query_number} Performance\n')
            f.write(f'- Documents query {query_number} brought to results pool: {doc_count:,}\n')
            f.write(f'- Number of matched documents from query {query_number}: {matched_doc_count}:\n')
            f.write(f'{matched_docs_from_query}\n')
            
    return count, total

            
def extract_evaluations(result_evaluations):
    # define the evaluation metrics to extract
    metrics = ["map", "P_20", "recall_20", "ndcg", "ndcg_cut_3", "ndcg_cut_5"]

    # split the evaluation string into a list of lines
    lines = result_evaluations.strip().split("\n")

    extracted_metrics = {}

    for line in lines:
        parts = line.split("\t")
        
        if len(parts) > 1:
            metric_name = parts[0].strip()

            if metric_name in metrics:

                metric_value = parts[-1].strip()
                extracted_metrics[metric_name] = metric_value

    return extracted_metrics


def select_turns_by_prefix(data, prefix):
    matched_turns = [turn for turn in data.get('turns', []) if turn.get('turn_id', '').startswith(prefix)]
    
    return matched_turns




def run_trec_eval(qrel_filename, output_filename, metric: str=None, metric_list: list =None, full_metrics: bool=False):
    # Define the path to the trec_eval executable
    trec_eval_path = './trec_eval/trec_eval'
    
    # Ensure the executable exists
    if not os.path.isfile(trec_eval_path):
        raise FileNotFoundError(f"The trec_eval executable was not found at {trec_eval_path}")

    if full_metrics:
        # Define the command and arguments
        command = [
            trec_eval_path,
            #'-q', 
            #'-c', 
            #'-M1000', 
            '-m',
            'all_trec',
            f'{settings.QRELS_PATH}/{qrel_filename}',
            output_filename
        ]
    
    elif metric_list is not None:
        # Define the command and arguments for multiple metrics
        command = [
            trec_eval_path,
        ]
        for single_metric in metric_list:
            command.extend(['-m', single_metric])
        command.extend([
            f'{settings.QRELS_PATH}/{qrel_filename}',
            output_filename
        ])
    
    elif metric is not None:
        # Define the command and arguments
        command = [
            trec_eval_path,
            '-m', 
            metric,
            f'{settings.QRELS_PATH}/{qrel_filename}',
            output_filename
            ]
        
    else:
        raise ValueError("Either full_metrics must be True or a metric (e.g. 'ndcg_cut.5') or metric_list must be provided.")

    
    # Use subprocess to run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Capture the output and error
    stdout, stderr = process.communicate()
    
    ## Check the return code
    if process.returncode != 0:
        print("Error running TREC Eval:")
        print(stderr)
        raise RuntimeError(f"trec_eval failed with return code {process.returncode}")
    
    return stdout


def extract_relevant_passages(passage_qrels, conversation_id, verbose_output_filename):
    # convert to np
    data = np.array([line.strip().split(' ') for line in passage_qrels])
    
    conversation_id = str(conversation_id)
    
    filtered_rows = data[data[:, 0] == conversation_id]

    result = []

    # process the filtered rows to get ranks 4 and 3
    rank_4_passages = []
    rank_3_passages = []
    for row in filtered_rows:
        rank = int(row[3])
        if rank == 4:
            rank_4_passages.append(row[2])
            result.append((rank, row[2]))
        elif rank == 3:
            rank_3_passages.append(row[2])
            result.append((rank, row[2]))

    # print all passages with rank 4
    if rank_4_passages:
        with open(verbose_output_filename, 'a') as f:
            f.write(f"\nPassages with rank 4:\n")
            f.write(f"\n{rank_4_passages}\n")
        
    
    # print all passages with rank 3
    if rank_3_passages:
        with open(verbose_output_filename, 'a') as f:
            f.write(f"\nPassages with rank 3:\n")
            f.write(f"\n{rank_3_passages}\n")


    # process the filtered rows to get rank 2 if the result has less than 20 items
    rank_2_passages = []
    if len(result) < 20:
        for row in filtered_rows:
            rank = int(row[3])
            if rank == 2:
                rank_2_passages.append(row[2])
                result.append((rank, row[2]))
                if len(result) >= 20:
                    break
    
    # print all passages with rank 2
    if rank_2_passages:
        with open(verbose_output_filename, 'a') as f:
            f.write(f"\nPassages with rank 2:\n")
            f.write(f"\n{rank_2_passages}\n")


    # process the filtered rows to get rank 1 if the result has less than 20 items
    rank_1_passages = []
    if len(result) < 20:
        for row in filtered_rows:
            rank = int(row[3])
            if rank == 1:
                rank_1_passages.append(row[2])
                result.append((rank, row[2]))
                if len(result) >= 20:
                    break

    # print all passages with rank 1
    if rank_1_passages:
        with open(verbose_output_filename, 'a') as f:
            f.write(f"\nPassages with rank 1:\n")
            f.write(f"\n{rank_1_passages}\n")

    # sort the result by rank in descending order
    result.sort(reverse=True, key=lambda x: x[0])
    
    # extract the passages from the sorted result
    sorted_passages = [passage for rank, passage in result]

    return sorted_passages



def passage_neural_retrieval(preliminary_hits, reranking_list, model, verbose_output_filename, query_weights=None, top_k=5):
        
    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # extract the raw documents from hits
    documents = [hit.lucene_document.get('raw') for hit in preliminary_hits]
    

    accumulated_scores = torch.zeros(len(documents), device=device)
    
    # assume equal weights for all queries if not provided
    if query_weights is None:
        query_weights = [1.0] * len(reranking_list)
    
    
    # ensure that the weights sum to 1
    query_weights = torch.tensor(query_weights, device=device)
    query_weights = query_weights / query_weights.sum()
    
    
    # iterate
    for m, model in enumerate(model):
        
        with open(verbose_output_filename, 'a') as f:
            f.write(f'\nModel {m + 1} ...\n')
            f.write(f'\nEmbedding the documents ...\n')
            
        # get the document embeddings for the current model
        current_document_embeddings = model.encode(documents, convert_to_tensor=True).to(device)
        
        for i, query in enumerate(reranking_list):
            # save all the prints to a text file for tracking system progress
            with open(verbose_output_filename, 'a') as f:
                f.write(f'\nReranking query {i + 1} ...\n')
            
            query_embedding = model.encode(query, convert_to_tensor=True).to(device)
            
            cosine_scores = util.cos_sim(query_embedding, current_document_embeddings)

            accumulated_scores += cosine_scores.squeeze(0) * query_weights[i]
    
    # average the accumulated scores
    total_iterations = len(reranking_list) * len(model)
    average_scores = accumulated_scores / total_iterations
    
    # find the top-k most similar documents based on the average scores
    top_results = torch.topk(average_scores, k=top_k)
    
    # extract
    top_indices = top_results.indices.tolist()    
    top_scores = top_results.values.tolist()
    
    # sort
    top_hits = []
    for rank, idx in enumerate(top_indices):
        hit = preliminary_hits[idx]
        hit.score = top_scores[rank] 
        top_hits.append(hit)
    
    return top_hits




