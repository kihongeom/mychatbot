import streamlit as st
import os

with st.sidebar:
    st.markdown("Author: [엄기홍](http://github.com/kihongeom)")
    st.markdown("이메일: [kheom@knu.ac.kr](mailto:kheom@knu.ac.kr)")
    


st.title("환영ㄴㄴ합니다")
st.markdown(
    """
"""
)

# OpenAI API 키 입력
api_key = st.text_input("OPENAI API 키 입력(ChatGPT): [발급 방법](https://wikidocs.net/233342)", type="password")

st.markdown(
    """
"""
)

# Tavily API 키 입력
search_api_key = st.text_input("Tavily Search API 키 입력(검색용, 선택): [발급 방법](https://wikidocs.net/234282)", type="password")
st.markdown(
    """
"""
)

# 설정 확인 버튼
confirm_btn = st.button("설정하기", key="api_key")


if confirm_btn:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.write(f"OPENAI API 키가 설정되었습니다: `{api_key[:15]}************`")

    if search_api_key:
        os.environ["TAVILY_API_KEY"] = search_api_key
        st.write(
            f"TAVILPY API 키가 설정되었습니다: `{search_api_key[:15]}************`"
        )
