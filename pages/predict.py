import streamlit as st

from utils.general import *
from utils.lib import *
from utils.predict_utils import *
from core.constants import *

st.header("ML Predictions")

# サイドバー
task = st.sidebar.selectbox("Select ML", TASK_LIST)
if task == TASK_LIST[0]:
    # 分類
    uploaded_model = st.sidebar.file_uploader("Upload ML model", type="pkl")
    uploaded_csv = st.sidebar.file_uploader("Upload csv file", type="csv")

    # メイン画面
    predict_classification(uploaded_csv, uploaded_model)

elif task == "回帰":
    uploaded_file = st.sidebar.file_uploader("ファイルをアップロード", type="csv")
elif task == "クラスタリング":
    st.write("Comming Soon")
elif task == "異常検出":
    st.write("Comming Soon")
elif task == "自然言語処理":
    st.write("Comming Soon")
elif task == "アソシエーション分析":
    st.write("Comming Soon")
