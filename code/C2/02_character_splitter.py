from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. 文档加载
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

# 2. 初始化固定大小分块器
text_splitter = CharacterTextSplitter(
    chunk_size=200,    # 每个块的大小
    chunk_overlap=10   # 块之间的重叠大小
)

# 3. 执行分块
chunks = text_splitter.split_documents(docs)
'''
先直接全部按照分隔符正则分割；
然后开始合并，合并至当前块内长度大于等于chunk_size，等于则直接结束当前块；大于则前面总和已经小于 chunk_size 的部分作为一块，新的段落开始 overlap 逻辑；
overlap: 不断减小刚结束的前一块（不断抛弃第一段落），
    条件1：确保前一块剩余段落不超过 chunk_overlap、
    条件2：确保块大小（前一块剩余段落 + 分隔符 + 上面还没入块新段落）不超过 chunk_size，
    循环结束时当前块（也就是添加了新段落下一块）就有了上一块的尾部 overlap 文本
'''

# 4. 打印结果
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
with open('../../data/C2/txt/chunks.txt', 'w', encoding='utf-8') as f:
    f.write('')
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
    with open('../../data/C2/txt/chunks.txt', 'a', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"\n')
