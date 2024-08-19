import streamlit as st
from rag import create_youtube_chain, rag_setup_vtt, download_auto_subtitles, convert_to_seconds, rag_setup_json
import os

st.subheader("유투브 요약 및 검색 챗봇")

if "OPENAI_API_KEY" in os.environ:
    st.markdown("<small> OpenAI API 키가 설정되었습니다.</small>", unsafe_allow_html=True)
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")

st.markdown(
    """ 

"""
)

# 대화 기록이 없다면, youtube_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "youtube_history" not in st.session_state or st.session_state["youtube_history"] is None:
    st.session_state["youtube_history"] = []

# chain 을 초기화
if "youtube_history" not in st.session_state or st.session_state["youtube_history"] is None:
    st.session_state["youtube_history"] = []

# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["youtube_history"].append({"role": role, "content": message})

# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["youtube_history"]:
        # role에 따라 메시지 출력 형식 결정
        if chat_message["role"] == "user":
            st.chat_message("user").write(chat_message["content"])
        else:
            st.chat_message("ai").write(chat_message["content"])

# 사이드 바
with st.sidebar:
    youtube_url = st.text_input("유투브 동영상  URL을 입력해주세요")
    
    # 설정이 완료 되었는지 확인하는 버튼
    confirm_btn = st.button("자막처리 시작")

# 자막 파일을 처리하는 함수
# @st.cache_resource(show_spinner="자막 파일을 처리 중입니다...")
def process_subtitle_file(url):
    file_path = download_auto_subtitles(url)
    # 다운로드된 파일을 사용하여 retriever를 설정
    retriever = rag_setup_json(file_path, chunk_size=300, chunk_overlap=50)
    return retriever

# 파일이 업로드 되었을 때
if confirm_btn:
    # youtube_url 긁어옴
    retriever = process_subtitle_file(youtube_url)
    st.session_state["youtube_chain"] = create_youtube_chain(retriever)

# 이전 까지의 대화를 출력
print_history()

user_input = st.chat_input("궁금한 내용을 입력해 주세요")

if user_input:
    # 파일이 업로드가 된 이후
    if st.session_state["youtube_chain"]:
        youtube_chain = st.session_state["youtube_chain"]
        # 사용자의 질문을 출력합니다.
        st.chat_message("user").write(user_input)

        # AI 의 답변을 출력합니다.
        with st.chat_message("ai"):
            # 답변을 출력할 빈 공간을 만든다.
            chat_container = st.empty()

            # 사용자가 질문을 입력하면, 체인에 질문을 넣고 실행합니다.
            answer = youtube_chain.stream(user_input)

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
        st.warning("유투브 링크를 입력해주세요.")
