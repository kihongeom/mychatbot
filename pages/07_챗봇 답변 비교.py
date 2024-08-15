import streamlit as st
import os
from rag import rag_setup2

st.subheader("챗봇 답변 결과 분석")

if "OPENAI_API_KEY" in os.environ:
    st.write("OpenAI API 키가 설정되었습니다.")
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")

st.markdown(
    """ 

"""
)

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# retreiver 초기화
if "my_retriever" not in st.session_state:
    st.session_state["my_retriever"] = None


with st.sidebar:
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

col1, col2 = st.columns(2)

search_k = col1.number_input(
    "검색할 문서의 개수", min_value=1, max_value=10, value=4, step=1
)

retriever_weight = col1.slider(
    "검색 가중치 (높음: 의미검색, 낮음: 키워드검색)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

chunk_size = col2.number_input(
    "Chunk Size", min_value=100, max_value=2000, value=500, step=100
)

chunk_overlap = col2.number_input(
    "Chunk Overlap", min_value=0, max_value=200, value=50, step=10
)

confirm_btn = st.button("설정 완료")


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


if confirm_btn:
    if uploaded_file:
        file_path = embed_file(uploaded_file)
        st.session_state["my_retriever"] = rag_setup2(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=search_k,
            weight=retriever_weight,
        )
        st.write("설정이 완료 되었습니다.")


# 채팅창 활성화
user_input = st.chat_input("검색하여 비교할 내용을 입력해 주세요")

if user_input:
    retriever = st.session_state["my_retriever"]
    if retriever:
        # 우리가 입력한 질문
        st.chat_message("user").write(user_input)

        # 문서 검색 후 결과 확인
        searched_documents = retriever.invoke(user_input)
        # List[Document]

        # AI 채팅 컨테이너 생성
        with st.chat_message("ai"):
            # 문서를 출력
            for i, doc in enumerate(searched_documents):
                st.markdown(f"[{i}] {doc.page_content}")
