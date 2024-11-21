# obtained from https://github.com/irlabamsterdam/iKAT/blob/main/2023/scripts/ikat_tools.py

# /path/to/hashes should point to a directory containing .tsv file(s) in the same format as mode 1 produces
# /path/to/passages should point to a directory containing .jsonl file(s) in the same format as mode 1 produces
# (these 2 paths can be the same directory)
# /path/to/log/file should be a filename where any hash mismatches will be logged If no mismatches
# are found the file will be empty. 

# to run
# python scripts/ikat_tools.py verify_hashes -H /path/to/hashes -c /path/to/passages -e /path/to/log/file



import csv
import hashlib
import json
import os
import sys
import tqdm

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings

# total number of passages that should result after segmenting with this script
PASSAGES = 116_838_987


# Paths
collection_hashes_path=F'{settings.COLLECTION_HASHES_PATH}'
collection_path=f'{settings.COLLECTION_PATH}'
errors_path='./logs/verify_hashes_log.txt'


"""
Verify a set of precomputed passage hashes against a set of passages.
"""
if not os.path.exists(collection_hashes_path):
    raise Exception(f'Hashes directory {collection_hashes_path} does not exist')

if not os.path.exists(collection_path):
    raise Exception(f'Collection directory {collection_path} does not exist')

existing_hashes = {}

hash_files = [f for f in os.listdir(collection_hashes_path) if f.endswith('.tsv')]
collection_files = [f for f in os.listdir(collection_path) if f.endswith('.jsonl')]

if len(hash_files) == 0:
    raise Exception(f'Failed to find any .tsv files in {collection_hashes_path}')

if len(collection_files) == 0:
    raise Exception(f'Failed to find any .jsonl files in {collection_path}')

print(f'> Will verify hashes from {len(hash_files)} .tsv files in {len(collection_files)} .jsonl files')

# read all the hashes from the .tsv file(s)
with tqdm.tqdm(desc='Reading hashes', total=PASSAGES) as pbar:
    for i, hash_file in enumerate(hash_files):
        reader = csv.reader(open(os.path.join(collection_hashes_path, hash_file), 'r'), delimiter='\t')
        for row in reader:
            clueweb_id, passage_id, passage_hash = row
            existing_hashes[f'{clueweb_id}:{passage_id}'] = passage_hash
            pbar.update(1)

errors = 0
# now scan through all the JSONL files, compute fresh hashes and compare to the existing ones
print(f'> Verifying hashes in {len(collection_path)} files')
with tqdm.tqdm(desc='Verifying hashes', total=PASSAGES) as pbar, open(errors_path, 'w') as error_file:
    for collection_file in collection_files:
        with open(os.path.join(collection_path, collection_file), 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break

                data = json.loads(line.strip())

                passage_id = data['id']
                assert(passage_id in existing_hashes)
                passage_content = data['contents']
                pmd5 = hashlib.md5()
                pmd5.update(passage_content.encode('utf-8'))
                computed_hash = pmd5.hexdigest()
                existing_hash = existing_hashes[passage_id]

                if computed_hash != existing_hash:
                    print(f'> ERROR: hash mismatch in {collection_file} on passage {passage_id}, computed hash {computed_hash}, existing hash {existing_hash}')
                    errors += 1
                    error_file.write(f'{collection_file},{passage_id},{computed_hash},{existing_hash}\n')

                pbar.update(1)

print(f'Hash verification finished with {errors} errors')


