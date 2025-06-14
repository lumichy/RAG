import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 1. サンプルドキュメント（faq_data_ja_500_dedup.jsonからQA形式で読み込み）
import json
with open("faq_data_ja_500_dedup.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)
docs = [f"Q: {item['question']} A: {item['answer']}" for item in faq_data]

# 2. テキスト分割
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_docs = text_splitter.create_documents(docs)

# 3. 埋め込みモデル（sentence-transformers/all-MiniLM-L6-v2）
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. ベクトルストア作成
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 5. Gemma 3 モデルのロード（google/gemma-3-1b-it）
model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

# pipelineを使わずに直接生成するカスタムLLM
class CustomGemma3LLM(LLM):
    def _call(self, prompt, stop=None, **kwargs):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        # attention_maskを明示的に作成
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
        # プロンプト部分を除去して生成部分のみ返す
        return decoded[len(prompt):].strip()
    @property
    def _llm_type(self):
        return "custom_gemma3"

llm = CustomGemma3LLM()

# 6. RAGチェーン構築
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 7. ユーザー入力で実行
if __name__ == "__main__":
    print("RAGシステムへようこそ。質問を入力してください。")
    while True:
        query = input("Q: ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        result = qa.invoke({"query": query})
        print("A:", result["result"] if isinstance(result, dict) and "result" in result else result)
