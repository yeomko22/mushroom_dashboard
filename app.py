import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from supabase import create_client

st.set_page_config(page_title="버섯 분류하기", page_icon="🍄")


st.markdown("""
<style>
header { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "scores" not in st.session_state:
    st.session_state.scores = None


@st.cache_resource
def init_supabase():
    return create_client(
        supabase_url=st.secrets["SUPABASE_URL"],
        supabase_key=st.secrets["SUPABASE_KEY"]
    )

supabase_client = init_supabase()


@st.cache_resource
def load_answer_df():
    return pd.read_csv("./data/mushroom_answer.csv")


def write_score(nickname: str, score: float):
    supabase_client.table("score").insert({
        "nickname": nickname,
        "score": score
    }).execute()


def read_score():
    response = (supabase_client.table("score")
                .select("*")
                .order("score", desc=True)
                .execute())
    return response.data


with st.spinner("데이터 로딩 중..."):
    answer_df = load_answer_df()
    scores = read_score()
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.drop(["id"], axis=1)
    scores_df["rank"] = [x for x in range(1, len(scores_df) + 1)]
    scores_df = scores_df.set_index(["rank"])
    scores_df['created_at'] = pd.to_datetime(scores_df['created_at']).dt.tz_convert('Asia/Seoul')
    scores_df["created_at"] = scores_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.scores = scores_df


st.title("버섯 분류하기 리더보드 🍄")
with st.form("form", clear_on_submit=True):
    col1, col2 = st.columns([0.2, 0.8])
    nickname = st.text_input("닉네임")
    submission = st.file_uploader(
        "결과 csv 파일 업로드",
        accept_multiple_files=False,
        type=['csv']
    )
    submit = st.form_submit_button("제출")

if submit:
    if not nickname:
        st.error("닉네임을 입력해주세요.")
        st.stop()
    if not submission:
        st.error("csv 파일을 제출해주세요.")
        st.stop()
    try:
        df = pd.read_csv(submission)
        if "mushroom_id" not in df.columns.to_list():
            raise ValueError("mushroom_id 컬럼이 비었습니다.")
        if "class" not in df.columns.to_list():
            raise ValueError("class 컬럼이 비었습니다.")
        if not answer_df["mushroom_id"].equals(df["mushroom_id"]):
            raise ValueError("유효하지 않은 mushroom id가 포함되어 있습니다.")
    except Exception as e:
        st.error(e)
        st.stop()

    score = accuracy_score(answer_df["class"], df["class"])
    st.markdown(f"현재 점수: {round(score, 4) * 100}%")
    write_score(nickname, score)
    st.rerun()


if st.session_state.scores is not None:
    st.write(st.session_state.scores)
