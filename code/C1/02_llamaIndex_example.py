import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 使用 AIHubmix
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )
local_model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "bge-small-zh-v1_5"
)
Settings.embed_model = HuggingFaceEmbedding(model_name=local_model_path)

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 从文档创建向量索引
# 内部会自动进行文本分块、嵌入计算和向量存储
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎
# 查询引擎用于执行用户查询并返回结果
query_engine = index.as_query_engine()

# 打印查询引擎使用的提示词模板
print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?").response)