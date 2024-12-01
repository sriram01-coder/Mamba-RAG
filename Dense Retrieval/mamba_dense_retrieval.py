import numpy as np
import pandas as pd 
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import time
import faiss
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

#model_name = 'meta-llama/Llama-3.1-8B-Instruct'
model_name = "tiiuae/falcon-mamba-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)
    
#model_args['attn_implementation'] = 'flash_attention_2'
# inputs = tokenizer("Hi how are you", return_tensors="pt").to(f"cuda:{model.device.index}")
# with torch.no_grad():
#     outputs = model(**inputs)
# exit()

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


NUM_TITLES = 10

class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=512)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

df_valid = pd.read_csv("/data6/sriramd/DLNLP/test.csv", index_col="id")

# Load embedding model
start = time.time()
print(f"Starting prompt embedding, t={time.time() - start :.1f}s")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda:0")

# Get embeddings of prompts
f = lambda row : " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]])
inputs = df_valid.apply(f, axis=1).values # better results than prompt only
#print(inputs[0])
# Each input in `inputs` is just the concatenation of the prompt + all the options. We are going to encode this text using the encoder.
prompt_embeddings = model.encode(inputs, show_progress_bar=False)

# Search closest sentences in the wikipedia index 
print(f"Loading faiss index, t={time.time() - start :.1f}s")
faiss_index = faiss.read_index("/data6/sriramd/DLNLP/faiss_index_bge-small-en-v1.5.index")
# faiss_index is the pre-encoded wikipedia index. We have encoded chunks of relevant wikipedia articles.

print(f"Starting text search, t={time.time() - start :.1f}s")
search_index = faiss_index.search(np.float32(prompt_embeddings), NUM_TITLES)[1]
# faiss_index.search allows us to search for the NUM_TITLES closest wikipedia chunks for a given prompt_embedding.
# So for every prompt_embedding, we get the indices of NUM_TITLES wikipedia chunks.
# faiss_index preserves the indices from the original wikipedia text of chunks. That is, text chunk i will have its encoding in index i in faiss_index. 

dataset = load_from_disk("/data6/sriramd/DLNLP/wikipedia")
# print(search_index[0])
# print(dataset[int(search_index[0][0])]["text"])
# for j in range(9, -1, -1):
#     print(f"{dataset[int(search_index[0][j])]['text']}\n")

df_valid = pd.read_csv("/data6/sriramd/DLNLP/test.csv", index_col="id")

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/data6/sriramd/Mamba",
        #attn_implementation="flash_attention_2"
        #quantization_config=bnb_config,
    )


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

def get_prompt_with_context(columns, context):
    input_prefix = f"Context: {context}Question: {columns[0]}\nA: {columns[1]}\nB: {columns[2]}\nC: {columns[3]}\nD: {columns[4]}\nE: {columns[5]}"
    instruction = f"Your tasks is to analyze the question and the given options below step by step. Pick the option that best answers the given question. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant. Answer with A, B, C, D or E only. "
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}\n\n### Response: "
    print("PROMPT = ", prompt)
    return prompt

def get_prompt_without_context(columns, context):
    input_prefix = f"Question: {columns[0]}\nA: {columns[1]}\nB: {columns[2]}\nC: {columns[3]}\nD: {columns[4]}\nE: {columns[5]}"
    instruction = f"Your tasks is to analyze the question and the given options below step by step. Pick the option that best answers the given question. Answer with A, B, C, D or E only. "
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}\n\n### Response: "
    return prompt

a_token_id = tokenizer.convert_tokens_to_ids("ĠA")
b_token_id = tokenizer.convert_tokens_to_ids("ĠB")
c_token_id = tokenizer.convert_tokens_to_ids("ĠC")
d_token_id = tokenizer.convert_tokens_to_ids("ĠD")
e_token_id = tokenizer.convert_tokens_to_ids("ĠE")

for num_retireved in [1, 3, 6, 10, 0]:
    predictions = []
    submit_ids = []
    start = time.time()
    for index in tqdm(range(df_valid.shape[0])):
        columns = df_valid.iloc[index].values
        submit_ids.append(columns[0])
        question = columns[1]
        options = [columns[2], columns[3], columns[4], columns[5], columns[6]]

        context = ""
        for j in range(num_retireved-1, -1, -1):
            context += f"{str(dataset[int(search_index[0][j])]['text'])}\n"
        
        if num_retireved != 0:
            inputs = tokenizer(get_prompt_with_context(columns, context), return_tensors="pt").to(f"cuda")
        else:
            inputs = tokenizer(get_prompt_without_context(columns, context), return_tensors="pt").to(f"cuda")
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
        print("output_string = ", output_string)
        exit()
        predictions.append(output_string)

    m = MAP_at_3(predictions, df_valid.answer.values)
    print('Num Retireved Documents = ', num_retireved, 'MAP@3 =', m)
    end = time.time()
    timer(start, end)