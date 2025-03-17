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
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

# 環境変数の読み込み
load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        self.parser = StrOutputParser()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.vector_store = None
        self.max_retrieval_attempts = 3  # 最大再試行回数

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
        attempt = state.get("attempt", 1)
        k = 3 + attempt  # 試行回数に応じて検索数を増やす

        docs = self.vector_store.similarity_search(question, k=k)
        return {
            "docs": docs,
            "question": question,
            "attempt": attempt
        }

    def evaluate_relevance(self, state: Dict) -> Dict:
        """検索結果の関連性を評価"""
        eval_prompt = ChatPromptTemplate.from_template("""
        質問と提供された文書の関連性を評価してください。

        質問:
        {question}

        文書:
        {context}

        0から100の数値で関連性を評価し、その理由も説明してください。
        回答は以下の形式で返してください：
        スコア: [数値]
        理由: [説明]
        """)

        context = "\n\n".join([doc.page_content for doc in state["docs"]])
        chain = eval_prompt | self.llm | self.parser
        response = chain.invoke({
            "question": state["question"],
            "context": context
        })

        # スコアを抽出（簡易的な実装）
        score_line = response.split('\n')[0]
        score = float(score_line.split(':')[1].strip())

        return {
            **state,
            "relevance_score": score,
            "evaluation": response
        }

    # ルーターからの条件付きエッジ
    def conditional_edge_score_retry(self, state: Dict) -> str:
        """次のステップを決定する"""
        score = state["relevance_score"]
        attempt = state["attempt"]

        if score < 80 and attempt < self.max_retrieval_attempts:
            print(f"score: {score}, attempt: {attempt}, 再施行します")
            return "retry_retrieve"
        else:
            print(f"score: {score}, attempt: {attempt}, 回答を生成します")
            return "final_answer"


    def generate_response(self, state: Dict) -> Dict:
        """回答の生成"""
        prompt = ChatPromptTemplate.from_template("""
        以下の情報を元に質問に答えてください。
        検索結果の関連性スコア: {score}/100

        情報:
        {context}

        質問:
        {question}

        回答:
        """)

        context = "\n\n".join([doc.page_content for doc in state["docs"]])
        chain = prompt | self.llm | self.parser
        response = chain.invoke({
            "context": context,
            "question": state["question"],
            "score": state["relevance_score"]
        })
        return {"response": response}

    def build_graph(self) -> StateGraph:
        """グラフの構築"""
        # Define the state schema
        from typing import TypedDict, List as ListType

        class State(TypedDict, total=False):
            question: str
            docs: ListType
            attempt: int
            relevance_score: float
            evaluation: str
            response: str

        workflow = StateGraph(State)

        # ノードの定義
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("evaluate", self.evaluate_relevance)
        workflow.add_node("generate", self.generate_response)

        # エッジの定義
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self.conditional_edge_score_retry,
            {
                "retry_retrieve": "retrieve",
                "final_answer": "generate"
            }
        )

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
        result = chain.invoke({
            "question": question,
            "attempt": 1
        })
        print("\n回答:", result["response"])
        print(f"\n検索精度: {result['relevance_score']}%")

if __name__ == "__main__":
    main()
