from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import bs4
from langchain_teddynote.prompts import load_prompt

def rag_setup(file_path, chunk_size=1000, chunk_overlap=50):
    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    # 반환값
    return retriever


def rag_setup2(file_path, chunk_size=1000, chunk_overlap=50, k=4, weight=0.5):
    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(
        # doc_list_1의 텍스트와 메타데이터를 사용하여 BM25Retriever를 초기화합니다.
        split_documents,
    )
    sparse_retriever.k = k  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

    # EnsembleRetriever로 dense_retriever와 sparse_retriever를 결합합니다.
    ensemble_retriever = EnsembleRetriever(
        # BM25Retriever와 FAISS retriever를 사용하여 EnsembleRetriever를 초기화하고, 각 retriever의 가중치를 설정합니다.
        retrievers=[dense_retriever, sparse_retriever],
        weights=[weight, 1 - weight],
    )

    # 반환값
    return ensemble_retriever


def naver_news_setup(url, chunk_size=1000, chunk_overlap=50):
    # 단계 1: 문서 로드(Load Documents)
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={
                    "class": ["newsct_article _article_body", "media_end_head_title"]
                },
            )
        ),
    )
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    # 반환값
    return retriever


def naver_news_crawling(url):
    # 단계 1: 문서 로드(Load Documents)
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={
                    "class": ["newsct_article _article_body", "media_end_head_title"]
                },
            )
        ),
    )
    docs = loader.load()

    return docs


def create_rag_chain(retriever, model_name="gpt-4o-mini"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = hub.pull("kihongeom/doc_summary")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def create_stuff_summary_chain():
    prompt = hub.pull("kihongeom/news_summary")

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        streaming=True,
    )

    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_chain


def create_rag_quiz_chain(retriever, model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """당신은 초중고 학생을 가르치는 30년차 베테랑 교사입니다.
    당신의 임무는 주어진 문맥(context)을 활용하여 학생들을 위한 퀴즈(quiz)를 만드는 것입니다.
    퀴즈는 4지선 다형 객관식 문제로 만들어 주세요. 문항은 3문항을 만들어 주세요.
    문항의 각 난이도는 쉬움, 보통, 어려움으로 나누어 주세요.
    
    #문제 예시:
    
    질문(난이도)
    - 가) 보기1
    - 나) 보기2
    - 다) 보기3
    - 라) 보기4
    
    - 정답:
    - 해설: 정답인 이유에 대해서 자세히 설명해 주세요.

    #Context: 
    {context}

    #Answer:"""
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
