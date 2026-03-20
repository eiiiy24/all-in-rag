from llama_index.core import VectorStoreIndex, Document, Settings, load_index_from_storage, StorageContext
from llama_index.core.base.embeddings.base import similarity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

local_model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "bge-small-zh-v1_5"
)
# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding(local_model_path)

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# 4. 加载索引
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
loader_index = load_index_from_storage(storage_context)

# 5. 执行查询
query = "LlamaIndex是做什么的？"
results = loader_index.as_retriever(similarity_top_k=1).retrieve(query)
for node in results:
    print(f"相似度分数: {node.score:.4f}")
    print(f"内容: {node.text}")
