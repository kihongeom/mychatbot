import streamlit as st
from langchain_core.messages import ChatMessage
from rag import naver_news_crawling, create_stuff_summary_chain
import os 
from datetime import datetime

st.subheader("네이버 뉴스 요약")

if "OPENAI_API_KEY" in os.environ:
    st.write("OpenAI API 키가 설정되었습니다.")
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")

st.markdown(
    """ 

"""
)

# 대화 기록이 없다면, summary_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "summary_history" not in st.session_state:
    st.session_state["summary_history"] = []

# chain 을 초기화
if "news_summary_chain" not in st.session_state:
    st.session_state["news_summary_chain"] = None

def create_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# 대화 기록에 채팅을 추가
def add_history(message):
    st.session_state["summary_history"].append(message)


# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["summary_history"]:
        # 이전에 요약한 내용 출력
        st.markdown(chat_message)


# 사이드 바
with st.sidebar:
    naver_news_url = st.text_input("네이버 뉴스 URL을 입력해주세요")

    # 설정이 완료 되었는지 확인하는 버튼
    confirm_btn = st.button("요약 시작")
    # 요약 기록을 지우는 버튼
    clear_history_btn = st.button("요약 지우기")


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="URL을 분석 중입니다...")
def embed_file(url):
    docs = naver_news_crawling(url)
    return docs


# 이전 까지의 대화를 출력
print_history()

# 파일이 업로드 되었을 때
if confirm_btn:
    # URL 긁어옴
    docs = embed_file(naver_news_url)
    # 요약 체인 생성
    summary_chain = create_stuff_summary_chain()
    # 사용자가 질문을 입력하면, 체인에 질문을 넣고 실행합니다.
    final_summary = summary_chain.invoke({"context": docs})
    st.markdown(f"##### 다음은 {naver_news_url} 의 요약본입니다.")
    # 요약본 출력
    st.markdown(final_summary)

    add_history(f"##### 다음은 {naver_news_url} 의 요약본입니다.")
    # 히스토리에 추가
    add_history(final_summary)

# 요약 기록을 지우는 버튼이 눌렸을 때
if clear_history_btn:
    st.session_state["summary_history"] = []
    st.success("요약이 모두 지워졌습니다.")
