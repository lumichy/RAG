import os
import sys
import json
import pdfplumber
import openpyxl
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # JSONの内容がリストかdictかで処理を分ける
    if isinstance(data, list):
        return "\n".join([json.dumps(item, ensure_ascii=False) for item in data])
    else:
        return json.dumps(data, ensure_ascii=False)

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n"
    return text

def extract_text_from_excel(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    text = ""
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
            text += row_text + "\n"
    return text

def collect_documents_from_folder(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            full_path = os.path.join(root, file)
            try:
                if ext == ".txt":
                    text = extract_text_from_txt(full_path)
                elif ext == ".json":
                    text = extract_text_from_json(full_path)
                elif ext == ".pdf":
                    text = extract_text_from_pdf(full_path)
                elif ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
                    text = extract_text_from_excel(full_path)
                else:
                    continue
                if text.strip():
                    docs.append(f"FILE: {file}\n{text}")
            except Exception as e:
                print(f"Failed to read {full_path}: {e}", file=sys.stderr)
    return docs

# 1. 指定フォルダからドキュメント収集
def get_docs_from_folder(folder):
    docs = collect_documents_from_folder(folder)
    if not docs:
        print("No valid documents found in the specified folder.", file=sys.stderr)
        sys.exit(1)
    return docs

# 2. テキスト分割
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.create_documents(docs)

# 3. 埋め込みモデル
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. ベクトルストア作成
def create_vectorstore(split_docs, embeddings):
    return FAISS.from_documents(split_docs, embeddings)

# 5. Gemma 3 モデルのロード
def load_llm():
    model_id = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    class CustomGemma3LLM(LLM):
        def _call(self, prompt, stop=None, **kwargs):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(model.device)
            attention_mask = (input_ids != tokenizer.eos_token_id).long()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            return decoded[len(prompt):].strip()
        @property
        def _llm_type(self):
            return "custom_gemma3"
    return CustomGemma3LLM()

# 6. RAGチェーン構築
def build_qa_chain(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", k=3)
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv main.py <folder_path>", file=sys.stderr)
        sys.exit(1)
    folder = sys.argv[1]
    print(f"指定フォルダからデータを読み込みます: {folder}")
    docs = get_docs_from_folder(folder)
    split_docs = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(split_docs, embeddings)
    llm = load_llm()
    qa = build_qa_chain(llm, vectorstore)
    print("RAGシステムへようこそ。質問を入力してください。")
    while True:
        query = input("Q: ")
        if query.strip().lower() in ["exit", "quit"]:
            break


        results_similarity = vectorstore.similarity_search(query, k=3)
        print("📄 類似内容:")
        for i, doc in enumerate(results_similarity):
            print(f"\n[{i}]\n{doc.page_content}")

        result = qa.invoke({"query": query})
        print("A:", result["result"] if isinstance(result, dict) and "result" in result else result)
