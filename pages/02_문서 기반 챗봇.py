import streamlit as st
from langchain_core.messages import ChatMessage
from rag import create_rag_chain, rag_setup2
import os
from datetime import datetime

st.subheader("문서 기반 챗봇")

if "OPENAI_API_KEY" in st.session_state:
    st.markdown("<small> OpenAI API 키가 설정되었습니다.</small>", unsafe_allow_html=True)
    # st.write("현재 세션에서 사용 중인 API 키:", st.session_state["OPENAI_API_KEY"])  # Debugging line
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")

st.markdown(
    """ 

"""
)

def create_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 대화 기록이 없다면, doc_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "doc_history" not in st.session_state:
    st.session_state["doc_history"] = []

# chain 을 초기화
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None


# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["doc_history"].append(ChatMessage(role=role, content=message))


# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["doc_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)


with st.sidebar:
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    retriever = rag_setup2(file_path, chunk_size=300, chunk_overlap=50)
    return retriever


# 파일이 업로드 되었을 때
if uploaded_file:
        # Check the file size
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
        st.write("파일이 너무 큽니다. 200MB 이하의 파일만 업로드할 수 있습니다.")
    else:
        try: 
            # st.write("File uploaded successfully!")  # Add this line for debugging
            retriever = embed_file(uploaded_file)
            st.session_state["rag_chain"] = create_rag_chain(retriever)
        except Exception as e:
            st.write("An error occurred while processing the file:", e)

# 이전 까지의 대화를 출력
print_history()

user_input = st.chat_input("궁금한 내용을 입력해 주세요")

if user_input:
    # 파일이 업로드가 된 이후
    if st.session_state["rag_chain"]:
        rag_chain = st.session_state["rag_chain"]
        # 사용자의 질문을 출력합니다.
        st.chat_message("user").write(user_input)

        # AI 의 답변을 출력합니다.
        with st.chat_message("ai"):
            # 답변을 출력할 빈 공간을 만든다.
            chat_container = st.empty()

            # 사용자가 질문을 입력하면, 체인에 질문을 넣고 실행합니다.
            answer = rag_chain.stream(user_input)

            # 스트리밍 출력
            ai_answer = ""
            for token in answer:
                ai_answer += token
                chat_container.markdown(ai_answer)

        # 대화 기록에 추가
        add_history("user", user_input)
        add_history("ai", ai_answer)
    else:
        # 파일이 업로드 되지 않았을 때
        st.warning("PDF 파일을 업로드 해주세요.")
