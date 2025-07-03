#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import ast
import re
from typing import List, Dict
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
load_dotenv()

# 1. 克隆源码
REPO_URL = "https://github.com/materialsproject/api.git"
REPO_DIR = "./api_repo"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# 2. 静态解析：遍历 .py 源文件，收集函数签名、文档、示例
records = []
for root, _, files in os.walk(REPO_DIR):
    for fname in files:
        if not fname.endswith(".py"): continue
        path = os.path.join(root, fname)
        src = open(path, encoding="utf8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sig = f"{node.name}{ast.get_source_segment(src, node.args)!s}"
                doc = ast.get_docstring(node) or ""
                # 简单用正则抽 Example 段
                m = re.search(r"```python(.*?)```", doc, flags=re.S)
                example = m.group(1).strip() if m else ""
                # 参数 & 返回部分解析（粗略 demo）
                params = [a.arg for a in node.args.args]
                # 尝试从 docstring 提取 Returns 部分
                returns = ""
                m_ret = re.search(r"Returns?:?\s*([\s\S]+?)(?:Args?:|Example:|$)", doc)
                if m_ret:
                    returns = m_ret.group(1).strip()
                records.append({
                    "file": path,
                    "func": node.name,
                    "signature": sig,
                    "doc": doc,
                    "params": params,
                    "returns": returns,
                    "example": example
                })

# 3. 输出部分解析结果（调试可用）
print(f"共解析 {len(records)} 个函数")
for r in records[:3]:
    print(r["func"], "| params:", r["params"], "returns:", r["returns"], "example:", bool(r["example"]))

# 4. 写为 JSON 文件，供后续检索
json_docs_dir = "./mp_docstore"
os.makedirs(json_docs_dir, exist_ok=True)
for idx, r in enumerate(records):
    with open(os.path.join(json_docs_dir, f"fn_{idx}.json"), "w", encoding="utf8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

# 5. 分字段构建向量索引
doc_texts = [r["doc"] for r in records]
param_texts = [", ".join(r["params"]) for r in records]
return_texts = [r["returns"] for r in records]

embedding = OpenAIEmbeddings(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model="text-embedding-3-small")

doc_store = Chroma.from_texts(doc_texts, embedding, persist_directory="./mp_index_doc")
param_store = Chroma.from_texts(param_texts, embedding, persist_directory="./mp_index_param")
return_store = Chroma.from_texts(return_texts, embedding, persist_directory="./mp_index_return")

print("分字段索引构建完成，可以用 doc_store/param_store/return_store 检索。")
