import streamlit as st
import os

# ###
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_teddynote import logging
# # 프로젝트 이름 변경은 .env에서 
# ###

# config는 상단에
st.set_page_config(
    page_title="나만을 위한 챗봇입니다",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://kihong.knu.ac.kr/',
        'Report a bug': "https://kihong.knu.ac.kr/",
        'About': "경북대학교 엄기홍 교수가 작성했습니다(https://kihong.knu.ac.kr/)"
    }
)

with st.sidebar:
    st.markdown("Author: [엄기홍](http://github.com/kihongeom)")
    st.markdown("이메일: [kheom@knu.ac.kr](mailto:kheom@knu.ac.kr)")
    st.markdown("※ 이 사이트의 이용과 관련된 책임은 사용자에게 있습니다.")

st.subheader("환영합니다")

# # 버튼 변수를 기본값으로 초기화
# api_confirm_btn = False
# search_confirm_btn = False


# API 키 처리
if "OPENAI_API_KEY" in os.environ:
    st.write("OpenAI API 키가 설정되었습니다.")
else:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    # OpenAI API 키 입력
    api_key = st.text_input("OPENAI API 키 입력(ChatGPT): [발급 방법](https://wikidocs.net/233342)", type="password")
    st.session_state['OPENAI_API_KEY'] = api_key
    api_confirm_btn = st.button("설정하기", key="api_key")
    
if "Tavily_API_KEY" in os.environ:
    st.write("Tavily API 키가 설정되었습니다.")
else:
    st.error("Tavily API 키가 설정되지 않았습니다.")
    # Tavily API 키 입력
    search_api_key = st.text_input("Tavily Search API 키 입력(검색용): [발급 방법](https://wikidocs.net/234282)", type="password")
    st.session_state['TAVILY_API_KEY'] = search_api_key
    search_confirm_btn = st.button("설정하기", key="search_api_key") 

# 설정 확인 버튼
if api_confirm_btn:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.write(f"OPENAI API 키가 설정되었습니다: `{api_key[:15]}************`")

if search_confirm_btn:
    if search_api_key:
        os.environ["TAVILY_API_KEY"] = search_api_key
        st.write(
            f"TAVILPY API 키가 설정되었습니다: `{search_api_key[:15]}************`"
        )

# # 설정 완료 여부 확인 및 안내 메시지 출력
# if api_confirm_btn or search_confirm_btn:
#     if not api_confirm_btn and "OPENAI_API_KEY" not in os.environ:
#         st.warning("OPENAI API 키를 설정해 주세요.")
#     if not search_confirm_btn and "Tavily_API_KEY" not in os.environ:
#         st.warning("Tavily Search API 키를 설정해 주세요.")
