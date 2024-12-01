import numpy as np
import pandas as pd 
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import time
import random

model_name = 'meta-llama/Llama-3.1-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)
    
#model_args['attn_implementation'] = 'flash_attention_2'

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/data6/sriramd/Mamba",
        #attn_implementation="flash_attention_2"
        #quantization_config=bnb_config,
    )

# inputs = tokenizer("Hi how are you", return_tensors="pt").to(f"cuda:{model.device.index}")
# with torch.no_grad():
#     outputs = model(**inputs)
# exit()

df_valid = pd.read_csv("/data6/sriramd/DLNLP/test.csv")

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

stop_words = ['each', 'you', 'the', 'use', 'used',
                  'where', 'themselves', 'nor', "it's", 'how', "don't", 'just', 'your',
                  'about', 'himself', 'with', "weren't", 'hers', "wouldn't", 'more', 'its', 'were',
                  'his', 'their', 'then', 'been', 'myself', 're', 'not',
                  'ours', 'will', 'needn', 'which', 'here', 'hadn', 'it', 'our', 'there', 'than',
                  'most', "couldn't", 'both', 'some', 'for', 'up', 'couldn', "that'll",
                  "she's", 'over', 'this', 'now', 'until', 'these', 'few', 'haven',
                  'of', 'wouldn', 'into', 'too', 'to', 'very', 'shan', 'before', 'the', 'they',
                  'between', "doesn't", 'are', 'was', 'out', 'we', 'me',
                  'after', 'has', "isn't", 'have', 'such', 'should', 'yourselves', 'or', 'during', 'herself',
                  'doing', 'in', "shouldn't", "won't", 'when', 'do', 'through', 'she',
                  'having', 'him', "haven't", 'against', 'itself', 'that',
                  'did', 'theirs', 'can', 'those',
                  'own', 'so', 'and', 'who', "you've", 'yourself', 'her', 'he', 'only',
                  'what', 'ourselves', 'again', 'had', "you'd", 'is', 'other',
                  'why', 'while', 'from', 'them', 'if', 'above', 'does', 'whom',
                  'yours', 'but', 'being', "wasn't", 'be']

def SplitList(mylist, chunk_size):
    return [mylist[offs:offs+chunk_size] for offs in range(0, len(mylist), chunk_size)]

def get_relevant_documents_parsed(df_valid):
    df_chunk_size=600
    paraphs_parsed_dataset = load_from_disk("/data6/sriramd/DLNLP/wikipedia")
    modified_texts = paraphs_parsed_dataset.map(lambda example:
                                             {'temp_text':
                                              f"{example['title']} {example['section']} {example['text']}".replace('\n'," ").replace("'","")},
                                             num_proc=2)["temp_text"]
    
    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx: idx+df_chunk_size]
    
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)
        
    article_indices_array =  np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)
    
    top_per_query = article_indices_array.shape[1]
    articles_flatten = [(
                         articles_values_array[index],
                         paraphs_parsed_dataset[idx.item()]["title"],
                         paraphs_parsed_dataset[idx.item()]["text"],
                        )
                        for index,idx in enumerate(article_indices_array.reshape(-1))]
    retrieved_articles = SplitList(articles_flatten, top_per_query)
    return retrieved_articles


def retrieval(df_valid, modified_texts):
    
    corpus_df_valid = df_valid.apply(lambda row:
                                     f'{row["prompt"]}\n{row["prompt"]}\n{row["prompt"]}\n{row["A"]}\n{row["B"]}\n{row["C"]}\n{row["D"]}\n{row["E"]}',
                                     axis=1).values
    vectorizer1 = TfidfVectorizer(ngram_range=(1,2),
                                 token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
                                 stop_words=stop_words)
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
                                 stop_words=stop_words,
                                 vocabulary=vocab_df_valid)
    vectorizer.fit(modified_texts[:500000])
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)
    
    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    chunk_size = 100000
    top_per_chunk = 10
    top_per_query = 10

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx: idx+chunk_size])
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)
    
    merged_top_scores = np.sort(top_values_array, axis=1)[:,-top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:,-top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]
    
    return articles_indices, merged_top_scores

def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U

m = MAP_at_3(['A B C', 'B C D'], ['A', 'B'])
print("MAP = ", m)


retrieved_articles_parsed = get_relevant_documents_parsed(df_valid)
gc.collect()

def get_prompt_with_context(columns, context):
    input_prefix = f"Context: {context}Question: {columns[1]}\nA: {columns[2]}\nB: {columns[3]}\nC: {columns[4]}\nD: {columns[5]}\nE: {columns[6]}"
    instruction = f"Your tasks is to analyze the question and the given options below step by step. Pick the option that best answers the given question. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant. Answer with A, B, C, D or E only."
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}\n\n### Response: "

    return prompt

a_token_id = tokenizer.convert_tokens_to_ids("ĠA")
b_token_id = tokenizer.convert_tokens_to_ids("ĠB")
c_token_id = tokenizer.convert_tokens_to_ids("ĠC")
d_token_id = tokenizer.convert_tokens_to_ids("ĠD")
e_token_id = tokenizer.convert_tokens_to_ids("ĠE")


for num_retireved in [1, 2, 5, 9]:
    predictions = []
    submit_ids = []
    start = time.time()
    for index in tqdm(range(df_valid.shape[0])):
        columns = df_valid.iloc[index].values
        submit_ids.append(columns[0])
        question = columns[1]
        options = [columns[2], columns[3], columns[4], columns[5], columns[6]]

        relevant_context = f"{retrieved_articles_parsed[index][-1][2]}\n"

        noisy_context = ""
        for j in range(num_retireved):
            rand_idx = random.randint(0, df_valid.shape[0] - 1)
            noisy_context += f"{retrieved_articles_parsed[rand_idx][-1][2]}\n"

        context = relevant_context + noisy_context

        inputs = tokenizer(get_prompt_with_context(columns, context), return_tensors="pt").to(f"cuda:{model.device.index}")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]

        # Apply softmax to get probabilities
        probs = torch.softmax(last_token_logits, dim=-1)
        a_prob = probs[a_token_id].item()
        b_prob = probs[b_token_id].item()
        c_prob = probs[c_token_id].item()
        d_prob = probs[d_token_id].item()
        e_prob = probs[e_token_id].item()

        # Define probabilities and their corresponding labels
        probabilities = [a_prob, b_prob, c_prob, d_prob, e_prob]
        labels = ['A', 'B', 'C', 'D', 'E']

        # Combine probabilities and labels into a list of tuples
        prob_with_labels = list(zip(probabilities, labels))

        # Sort the tuples by probability in descending order
        sorted_prob_with_labels = sorted(prob_with_labels, key=lambda x: x[0], reverse=True)

        # Extract the top 3 labels
        top_3_labels = [label for _, label in sorted_prob_with_labels[:3]]

        # Form the output string
        output_string = " ".join(top_3_labels)

        predictions.append(output_string)

    m = MAP_at_3(predictions, df_valid.answer.values)
    print('Num Retireved Documents = ', num_retireved, 'MAP@3 =', m)
    end = time.time()
    timer(start, end)