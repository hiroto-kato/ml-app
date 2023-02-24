import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.learn_utils import *
from core.constants import *

st.header("ML Learning")

# サイドバー
task = st.sidebar.selectbox("Select ML", TASK_LIST)
if task == TASK_LIST[0]:
    # 分類
    uploaded_file = st.sidebar.file_uploader("Upload csv file", type="csv")
    is_sample_data = st.sidebar.checkbox(
        "Sample data", value=False
    )  # uploaded_fileある時はFalseにしたい

    # メイン画面
    main_classification(uploaded_file, is_sample_data)

elif task == TASK_LIST[1]:
    # 回帰
    uploaded_file = st.sidebar.file_uploader("Upload csv file", type="csv")
    is_sample_data = st.sidebar.checkbox(
        "Sample data", value=False
    )  # uploaded_fileある時はFalseにしたい

    # メイン画面
    main_regression(uploaded_file, is_sample_data)

elif task == "クラスタリング":
    st.write("Comming Soon")
elif task == "異常検出":
    st.write("Comming Soon")
elif task == "自然言語処理":
    st.write("Comming Soon")
elif task == "アソシエーション分析":
    st.write("Comming Soon")
