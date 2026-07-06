import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="나만의 AI 조수", page_icon="🤖")
st.title("🤖 나만의 AI 조수")

# ---------------------------------------------------------
# API 키 가져오기
# 1순위: Streamlit의 "Secrets"에 저장된 키 (배포할 때 사용, 안전함)
# 2순위: 화면에서 직접 입력 (로컬 테스트용)
# ---------------------------------------------------------
api_key = st.secrets.get("NVIDIA_API_KEY", None)

if not api_key:
    api_key = st.sidebar.text_input("NVIDIA API 키를 입력하세요", type="password")

if not api_key:
    st.info("왼쪽 사이드바에 API 키를 먼저 입력해주세요.")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

SYSTEM_PROMPT = "너는 친절하고 똑똑한 한국어 AI 비서야. 명확하고 도움이 되게 답변해줘."

# 대화 기록 저장 (새로고침하기 전까지 유지됨)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 지금까지의 대화 화면에 보여주기 (system 메시지는 숨김)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 사용자 입력창
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 스트리밍으로 받기
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model="z-ai/glm-5.2",
                messages=st.session_state.messages,
                max_tokens=16384,
                temperature=1.0,
                top_p=1.0,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"오류가 발생했습니다: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
