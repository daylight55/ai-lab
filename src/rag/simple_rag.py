from typing import Dict, List, Annotated
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph
from langchain_anthropic import ChatAnthropic

# 環境変数の読み込み
load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.vector_store = None

    def load_documents(self):
        # miscディレクトリからPDFファイルを読み込む
        pdf_dir = Path("misc")
        pdf_files = list(pdf_dir.glob("*.pdf"))

        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        # テキストを分割
        splits = self.text_splitter.split_documents(documents)

        # Chromaベクトルストアの作成
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )

    def retrieve(self, state: Dict) -> Dict:
        """関連文書の検索"""
        question = state["question"]
        docs = self.vector_store.similarity_search(question, k=3)
        return {"docs": docs, "question": question}

    def generate_response(self, state: Dict) -> Dict:
        """回答の生成"""
        prompt = ChatPromptTemplate.from_template("""
        以下の情報を元に質問に答えてください。

        情報:
        {context}

        質問:
        {question}

        回答:
        """)

        context = "\n\n".join([doc.page_content for doc in state["docs"]])
        response = self.llm.invoke(
            prompt.format(context=context, question=state["question"])
        )

        return {"response": response.content}

    def build_graph(self) -> Graph:
        """グラフの構築"""
        workflow = Graph()

        # ノードの定義
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate_response)

        # エッジの定義
        workflow.add_edge("retrieve", "generate")

        # 入力と出力の設定
        workflow.set_entry_point("retrieve")
        workflow.set_finish_point("generate")

        return workflow

def main():
    # RAGパイプラインの初期化
    rag = RAGPipeline()
    rag.load_documents()

    # グラフの構築
    graph = rag.build_graph()
    chain = graph.compile()

    # 対話ループ
    print("PDFに関する質問にお答えします。終了するには 'quit' と入力してください。")
    while True:
        question = input("\n質問を入力してください: ")
        if question.lower() == 'quit':
            break

        # 質問の実行
        result = chain.invoke({"question": question})
        print("\n回答:", result["response"])

if __name__ == "__main__":
    main()
