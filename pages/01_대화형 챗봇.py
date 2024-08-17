import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from datetime import datetime
from langchain_teddynote import logging
import os 


st.subheader("나만의 챗봇")

if "OPENAI_API_KEY" in os.environ:
    st.markdown("<small> OpenAI API 키가 설정되었습니다.</small>", unsafe_allow_html=True)
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")

st.markdown(
    """ 

"""
)

def create_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


# 대화 기록이 없다면, chat_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))


def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


with st.sidebar:
    session_id = st.text_input("Conversation Session ID", "Session ID 입력하세요")


def create_chain(model_name="gpt-4o-mini"):
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "As a seasoned journalist, your task is to structure an article on [topic]. The article should be well-researched, factual, and engaging to a diverse readership. Start with an attention-grabbing introduction that provides context and piques the reader's interest. Organize the content into coherent sections, each with a clear focus, and use compelling headlines for each section. Incorporate quotes from reputable sources, and always include citations or references for any data, quotes, or analysis provided. Provide in-depth analysis and perspectives on the topic. Conclude the article with a thought-provoking summary that encourages further reflection or action. Ensure the article adheres to journalistic ethics and standards, and is suitable for publication in a respected news outlet. If you do not know the answer to a particular question or cannot find sufficient information, clearly state that you do not know rather than speculating"),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    # 대화를 기록하는 체인
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    # 완성된 체인을 반환
    return chain_with_history


# 이전의 대화 내용들을 출력
print_history()

# 사용자가 입력할 상황
question = st.chat_input("대화를 시작해 보세요!")

# request_btn 클릭이 된다면...
if question:
    # 1) 사용자가 입력한 상황을 먼저 출력
    st.chat_message("user").write(question)

    # AI가 답변을 생성하기 위한 Chain 을 생성
    chain = create_chain('gpt-4o-mini')

    answer = chain.stream(
        # 질문 입력
        {"question": question},
        # 세션 ID 기준으로 대화를 기록합니다.
        config={"configurable": {"session_id": session_id}},
    )

    with st.chat_message("ai"):
        # 스트리밍 답변을 출력할 빈 컨테이너를 만든다.
        chat_container = st.empty()

        # AI 답변
        ai_answer = ""
        for token in answer:
            # 토큰 단위로 실시간 출력한다.
            ai_answer += token
            chat_container.markdown(ai_answer)

    # 대화 내용을 기록에 추가하는 내용
    add_history("user", question)  # 사용자의 질문
    add_history("ai", ai_answer)  # 챗봇의 답변
