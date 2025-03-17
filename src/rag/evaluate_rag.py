"""
RAG (Retrieval-Augmented Generation) パイプラインの評価モジュール

このモジュールは、PDFドキュメントからの情報検索と生成を行うRAGパイプラインを実装します。
検索結果の関連性を評価し、必要に応じて再検索を行うことで、より質の高い回答を生成します。
LangChain、LangGraph、Chroma、およびAnthropicのClaudeモデルを使用しています。

使用方法:
    python -m src.rag.evaluate_rag

環境変数:
    OPENAI_API_KEY: OpenAIのAPIキー（埋め込み生成用）
    ANTHROPIC_API_KEY: AnthropicのAPIキー（テキスト生成用）
"""

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
    """
    RAG (Retrieval-Augmented Generation) パイプラインを実装するクラス

    このクラスは、PDFドキュメントからの情報検索、検索結果の評価、
    および質問に対する回答生成の一連のプロセスを管理します。
    LangGraphを使用して、検索→評価→（必要に応じて再検索）→回答生成
    というワークフローを構築します。
    """
    def __init__(self):
        """
        RAGPipelineクラスの初期化

        以下のコンポーネントを初期化します：
        - OpenAI埋め込みモデル（ドキュメントとクエリの埋め込み用）
        - Anthropicの言語モデル（回答生成用）
        - テキスト分割器（ドキュメントを適切なサイズのチャンクに分割）
        - 出力パーサー（LLMの出力を処理）

        ベクトルストアは後でload_documents()メソッドで初期化されます。
        """
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
        """
        PDFドキュメントを読み込み、ベクトルストアを作成します

        miscディレクトリ内のすべてのPDFファイルを読み込み、
        テキスト分割器でチャンクに分割し、
        OpenAI埋め込みを使用してChromaベクトルストアを作成します。

        Returns:
            None
        """
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
        """
        質問に関連する文書を検索します

        質問に対して類似度検索を実行し、関連する文書を取得します。
        試行回数が増えるごとに、より多くの文書を取得します。

        Args:
            state: 現在の状態を含む辞書
                - question: ユーザーの質問
                - attempt: 現在の試行回数（デフォルト: 1）

        Returns:
            Dict: 更新された状態を含む辞書
                - docs: 検索された文書のリスト
                - question: 元の質問
                - attempt: 現在の試行回数
        """
        question = state["question"]
        attempt = state.get("attempt", 1)
        k = 3 + attempt  # 試行回数に応じて検索数を増やす

        docs = self.vector_store.similarity_search(question, k=k)
        return {"docs": docs, "question": question, "attempt": attempt}

    def evaluate_relevance(self, state: Dict) -> Dict:
        """
        検索結果の関連性をLLMを使用して評価します

        検索された文書が質問にどの程度関連しているかを、
        0から100のスコアで評価します。このスコアは後で
        再検索するかどうかの判断に使用されます。

        Args:
            state: 現在の状態を含む辞書
                - question: ユーザーの質問
                - docs: 検索された文書のリスト

        Returns:
            Dict: 更新された状態を含む辞書
                - relevance_score: 関連性スコア（0-100）
                - evaluation: 評価の詳細な説明
                - その他、入力状態のすべての要素
        """
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
        response = chain.invoke({"question": state["question"], "context": context})

        # スコアを抽出（簡易的な実装）
        score_line = response.split("\n")[0]
        score = float(score_line.split(":")[1].strip())

        return {**state, "relevance_score": score, "evaluation": response}

    def conditional_edge_score_retry(self, state: Dict) -> str:
        """
        次のステップを決定するルーター関数

        関連性スコアと試行回数に基づいて、再検索するか
        最終回答を生成するかを決定します。

        Args:
            state: 現在の状態を含む辞書
                - relevance_score: 関連性スコア（0-100）
                - attempt: 現在の試行回数

        Returns:
            str: 次のステップを示す文字列
                - "retry_retrieve": 再検索を行う
                - "final_answer": 最終回答を生成する
        """
        score = state["relevance_score"]
        attempt = state["attempt"]

        if score < 80 and attempt < self.max_retrieval_attempts:
            print(f"score: {score}, attempt: {attempt}, 再施行します")
            return "retry_retrieve"
        else:
            print(f"score: {score}, attempt: {attempt}, 回答を生成します")
            return "final_answer"

    def generate_response(self, state: Dict) -> Dict:
        """
        検索結果に基づいて質問に対する回答を生成します

        検索された文書の内容を使用して、ユーザーの質問に対する
        回答をLLMを使って生成します。関連性スコアも提供して、
        LLMが回答の確信度を調整できるようにします。

        Args:
            state: 現在の状態を含む辞書
                - question: ユーザーの質問
                - docs: 検索された文書のリスト
                - relevance_score: 関連性スコア

        Returns:
            Dict: 生成された回答を含む辞書
                - response: 生成された回答テキスト
        """
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
        response = chain.invoke(
            {
                "context": context,
                "question": state["question"],
                "score": state["relevance_score"],
            }
        )
        return {"response": response}

    def build_graph(self) -> StateGraph:
        """
        RAGパイプラインのワークフローグラフを構築します

        LangGraphを使用して、検索→評価→（必要に応じて再検索）→回答生成
        というワークフローを定義します。状態の型定義、ノードの追加、
        エッジの定義、条件付きエッジの設定を行います。

        Returns:
            StateGraph: 構築されたワークフローグラフ
        """
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
            {"retry_retrieve": "retrieve", "final_answer": "generate"},
        )

        # 入力と出力の設定
        workflow.set_entry_point("retrieve")
        workflow.set_finish_point("generate")

        return workflow


def main():
    """
    メイン関数 - RAGパイプラインを初期化し、対話ループを実行します

    PDFドキュメントを読み込み、ワークフローグラフを構築し、
    ユーザーからの質問に対して回答を生成する対話ループを実行します。
    ユーザーが 'quit' と入力するまで続けます。

    Returns:
        None
    """
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
        if question.lower() == "quit":
            break

        # 質問の実行
        result = chain.invoke({"question": question, "attempt": 1})
        print("\n回答:", result["response"])
        print(f"\n検索精度: {result['relevance_score']}%")


if __name__ == "__main__":
    main()
