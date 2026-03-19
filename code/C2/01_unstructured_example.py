from test_unstructured.partition.html.test_convert import elements
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    ocr_languages="chi_sim"
)
# elements = partition_pdf(
#     filename=pdf_path,
#     content_type="application/pdf",
#     strategy="hi_res", # 连接 huggingface 报错就改 huggingface_hub 源码 constant.py
#     ocr_languages="chi_sim",
#     # 报错 PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH? 安装 poppler .\poppler-25.12.0\Library\bin加入系统环境变量
#     # 报错 TesseractNotFoundError: tesseract is not installed or it's not in your PATH. 安装tesseract .\Tesseract-OCR 加入系统环境变量
# )
# elements = partition_pdf(
#     filename=pdf_path,
#     strategy="ocr_only",
#     ocr_languages="chi_sim"
#     # 报错 NoModule 就 pip install
# )

# 补充：下载中文语言包 chi_sim.traineddata 到 ..\Tesseract-OCR\tessdata

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)