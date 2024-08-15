import streamlit as st
from langchain_core.messages import ChatMessage
from 사용하지_도구.agent1 import create_agent
import os

st.subheader("Tavily 검색 챗봇")

if "Tavily_API_KEY" in os.environ:
    st.write("Tavily API 키가 설정되었습니다.")
else:
    st.error("Tavily API 키가 설정되지 않았습니다.")

st.markdown(
    """ 

"""
)

# 대화 기록이 없다면, chat_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# chain 을 초기화
if "search_agent" not in st.session_state:
    st.session_state["search_agent"] = None


# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))


# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)


# 사이드 바
with st.sidebar:
    search_count = st.number_input(
        "검색 결과의 개수를 입력해주세요", min_value=1, max_value=10, value=5, step=1
    )

    # 설정이 완료 되었는지 확인하는 버튼
    confirm_btn = st.button("설정 완료")
    st.markdown("검색 결과 **개수**를 설정하신 후 검색해 주세요")

# 파일이 업로드 되었을 때
if confirm_btn:
    st.session_state["search_agent"] = create_agent(k=search_count)

# 이전 까지의 대화를 출력
print_history()

user_input = st.chat_input("궁금한 내용을 입력해 주세요")

if user_input:
    # 파일이 업로드가 된 이후
    if st.session_state["search_agent"]:
        search_agent = st.session_state["search_agent"]
        # 사용자의 질문을 출력합니다.
        st.chat_message("user").write(user_input)

        # AI 의 답변을 출력합니다.
        with st.chat_message("ai"):

            # 사용자가 질문을 입력하면, 체인에 질문을 넣고 실행합니다.
            ai_answer = st.session_state["search_agent"].invoke({"input": user_input})

            st.write(ai_answer["output"])

        # 대화 기록에 추가
        add_history("user", user_input)
        add_history("ai", ai_answer["output"])
