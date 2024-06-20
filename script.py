from sentence_transformers import SentenceTransformer
import pandas as pd
from pandas import read_csv
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, Index
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import random
from collections import defaultdict
import massedit
filenames = ['jobpostings.csv']
massedit.edit_files(filenames, ["re.sub(r'\"\",\"\"', ',', line)"], dry_run=True)

my_delimiter = "\",\""

df = pd.read_csv(
    filepath_or_buffer='jobpostings.csv',
    delimiter=my_delimiter,
    quoting=3,
    on_bad_lines='error',
    engine='python',
    usecols = [1,3,4]
)

str_df = df.astype(str).values.tolist()

df['sentence'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
sentences = df['sentence'].tolist()
shortened_sentences = sentences[:100]

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

encoded_input = tokenizer(shortened_sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    model_output = model(**encoded_input)

token_embeddings = model_output.last_hidden_state
#print(f"Token embeddings shape: {token_embeddings.size()}")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
# Normalize the embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#print(f"Sentence embeddings shape: {sentence_embeddings.size()}")

sentence_embeddings = sentence_embeddings.detach().numpy()

scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))

for idx in range(sentence_embeddings.shape[0]):
    scores[idx, :] = cosine_similarity([sentence_embeddings[idx]], sentence_embeddings)[0]

##########################

# Load the embeddings
embeddings = sentence_embeddings

# Connect to Milvus

# Disconnect the existing connection with alias 'default'
connections.disconnect("default")

#This command is for running two separate contianers
#docker run -it --network host krishgangal_script 
#connections.connect("default", host="localhost", port="19530")

#This command is for running everything from ONE docker compose
connections.connect("default", host="host.docker.internal", port="19530")


# Define a schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1])
]
schema = CollectionSchema(fields, description="Embedding collection")

# Create a collection
collection_name = "embedding_collection"
collection = Collection(name=collection_name, schema=schema)

# Insert the embeddings
data = [
    [i for i in range(len(embeddings))],  # IDs
    embeddings.tolist()  # Embeddings
]
collection.insert(data)

# Create an index for the embeddings
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 128}
}
index = Index(collection, "embedding", index_params)

# Load the collection to memory
collection.load()

# Define a method to search for duplicates
def search_duplicates(collection, query_embeddings, threshold=0.9, top_k=10):
    results = collection.search(
        data=query_embeddings,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=None
    )
    
    duplicates = []
    for i, result in enumerate(results):
        for hit in result:
            if hit.distance >= threshold:
                duplicates.append((i, hit.id, hit.distance))
    
    return duplicates

# Clustering using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
labels = dbscan.fit_predict(embeddings)

# Inspect some clusters
for label in set(labels):
    if label != -1:  # -1 is noise in DBSCAN
        #print(f"Cluster {label}:")
        cluster_indices = np.where(labels == label)[0]
        #print(cluster_indices)

# Compute the silhouette score
score = silhouette_score(embeddings, labels, metric='cosine')
#print(f"Silhouette Score: {score}")

# Check nearest neighbors consistency
def check_nearest_neighbors_consistency(collection, embeddings, top_k=5):
    results = collection.search(
        data=embeddings,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=None
    )
    
    consistent_neighbors = 0
    for i, result in enumerate(results):
        neighbor_ids = [hit.id for hit in result]
        if i in neighbor_ids:
            consistent_neighbors += 1
    
    consistency_rate = consistent_neighbors / len(embeddings)
    return consistency_rate

consistency_rate = check_nearest_neighbors_consistency(collection, embeddings)
#print(f"Nearest Neighbors Consistency Rate: {consistency_rate}")

# Manual inspection
def manual_inspection(duplicates, num_samples=10):
    sample_indices = random.sample(range(len(duplicates)), num_samples)
    for index in sample_indices:
        query_id, duplicate_id, distance = duplicates[index]
        #print(f"Query ID: {query_id}, Duplicate ID: {duplicate_id}, Distance: {distance}")

# Load query embeddings (for example, the same embeddings used for insertion)
query_embeddings = embeddings[:10]  # Take the first 10 for testing

# Search for duplicates
duplicates = search_duplicates(collection, query_embeddings)

manual_inspection(duplicates)

# Adjusting Threshold
thresholds = [0.95, 0.9, 0.85, 0.8]
for threshold in thresholds:
    duplicates = search_duplicates(collection, embeddings[:10], threshold=threshold)
    #print(f"Threshold: {threshold}, Number of duplicates found: {len(duplicates)}")


# Load the collection
collection_name = "embedding_collection"
collection = Collection(name=collection_name)

# Define a method to search for duplicates
def search_duplicates(collection, embeddings, threshold=0.9, top_k=10):
    results = collection.search(
        data=embeddings,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        expr=None
    )
    
    duplicates = defaultdict(list)
    for i, result in enumerate(results):
        for hit in result:
            if hit.distance >= threshold:
                duplicates[i].append(hit.id)
    
    return duplicates

# Search for duplicates
duplicates = search_duplicates(collection, embeddings, threshold=0.9, top_k=10)

# Output duplicates

print("Duplicate Rows:")
for query_id, duplicate_ids in duplicates.items():
    if len(duplicate_ids) > 1:
        print(query_id + 2)
    #print(f"Document ID {query_id} is a duplicate of Document IDs {duplicate_ids}")

