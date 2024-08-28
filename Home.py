import streamlit as st

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

# Check and Set API Key
if "OPENAI_API_KEY" not in st.session_state:
    api_key = st.text_input("OPENAI API 키 입력(ChatGPT): [발급 방법](https://wikidocs.net/233342)", type="password")
    if st.button("설정하기", key="api_key"):
        if api_key:
            st.session_state["OPENAI_API_KEY"] = api_key
            st.success("OPENAI API 키가 설정되었습니다.")
            # st.write("API 키 설정 완료:", st.session_state["OPENAI_API_KEY"])  # Debugging line
        else:
            st.error("유효한 API 키를 입력해 주세요.")
else:
    st.markdown("<small> OpenAI API 키가 설정되었습니다.</small>", unsafe_allow_html=True)
    # st.write("현재 세션에서 사용 중인 API 키:", st.session_state["OPENAI_API_KEY"])  # Debugging line

