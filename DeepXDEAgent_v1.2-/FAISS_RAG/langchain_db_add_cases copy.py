import os
import re
import uuid
from langchain.docstore.document import Document  # 使用内置的 Document 类
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
from huggingface_hub import hf_hub_download

def load_config(file_path):
    # global config
    with open(file_path, 'r', encoding='utf-8') as file:
        print(f"reading file {file_path}...")
        content = json.load(file)
    return content

print(os.getcwd())
print(os.path.dirname(__file__))
print(os.path.dirname(os.getcwd()))
print(os.path.abspath("./FAISS_save.py"))

prj_root = os.path.dirname(os.path.dirname(__file__))
print(prj_root)

database_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
documents_pde_problem = []
documents_pde_case = []

# 保存 PDE 问题
batch_size = 5
# hfe_model_name = "tbs17/MathBERT"
hfe_model_name = "sentence-transformers/all-mpnet-base-v2"


# TODO: 路径不能有中文
# persist_pde_problem = os.path.join(database_root, "pde_problem_faiss")
# persist_pde_case = os.path.join(database_root, "pde_case_faiss")
persist_pde_problem = "../database/pde_problem_faiss"
persist_pde_case = "../database/pde_case_faiss"


# 加载 PDE 问题
pde_problem_vectordb_loaded = FAISS.load_local(
    persist_pde_problem, 
    HuggingFaceEmbeddings(model_name=hfe_model_name),
    allow_dangerous_deserialization=True)
pde_case_vectordb_loaded = FAISS.load_local(
    persist_pde_case, 
    HuggingFaceEmbeddings(model_name=hfe_model_name),
    allow_dangerous_deserialization=True)

# 测试
# 这会 pop 掉一个
key, value = pde_case_vectordb_loaded.docstore._dict.popitem()
tmp1 = value.page_content
tmp1_content = json.loads(tmp1)
print(tmp1_content)
with open(os.path.join(database_root, "tmp1.json"), "w", encoding="utf-8") as f:
    json.dump(tmp1_content, f, ensure_ascii=False, indent=4)
    
results = pde_problem_vectordb_loaded.similarity_search(query="", k=1,filter={"uid": value.metadata['uid']})
print(results)
tmp2 = results[0].page_content
# tmp2 = json.loads({"problem_description": tmp2})
print(tmp2)
with open(os.path.join(database_root, "tmp2.json"), "w", encoding="utf-8") as f:
    json.dump({"problem_description": tmp2}, f, ensure_ascii=False, indent=4)

print(hf_hub_download(repo_id=hfe_model_name, filename="config.json"))
