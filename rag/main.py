# Modified from https://rocm.blogs.amd.com/artificial-intelligence/rag-llamaindex/README.html

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts.base import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import BeautifulSoupWebReader

import time

def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"


llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"use_safetensors": False},
    # tokenizer_kwargs={},
    generate_kwargs={"do_sample":True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# global setting
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=32)
#Settings.num_output = 512
#Settings.context_window = 3900

url = "https://paulgraham.com/hwh.html"
documents = BeautifulSoupWebReader().load_data([url])

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=8)

question = "How does Paul Graham recommend to work hard? Can you list it as steps"

print("warmup..")
for _ in range(3):
    response = query_engine.query(question)
print(response)

print("benchmarking..")
start = time.perf_counter()

for _ in range(10):
    _ = query_engine.query(question)

end = time.perf_counter()

print(f"Total elapsed time of 10 queries: {end - start:0.4f} seconds")
