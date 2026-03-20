import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
import nest_asyncio
from llama_index.core.response.notebook_utils import display_source_node
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

documents = SimpleDirectoryReader(input_files=["../../data/C6/paul_graham/paul_graham_essay.txt"]).load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

vector_index = VectorStoreIndex(nodes)
retriever = vector_index.as_retriever(similarity_top_k=2)

retrieved_nodes = retriever.retrieve("What did the author do growing up?")
for node in retrieved_nodes:
    display_source_node(node, source_length=1000)