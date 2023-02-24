import os
import io
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pycaret import classification, regression

from utils.lib import *


def predict_classification(uploaded_csv: UploadedFile, uploaded_model: UploadedFile):
    """分類の予測する処理

    Parameters
    ----------
    uploaded_csv : UploadedFile
        csvファイル
    uploaded_model : UploadedFile
        pklファイル
    """

    if uploaded_csv is not None and uploaded_model is not None:
        st.subheader("Input data")
        df = pd.read_csv(uploaded_csv)
        st.write(df)
        start = st.sidebar.button("Start")
        if start:
            # ファイルを一時保存してモデルを読み込む
            with open(uploaded_model.name, "wb") as f:
                f.write(uploaded_model.getvalue())
            model = classification.load_model(os.path.splitext(uploaded_model.name)[0])

            # 一時保存したファイルを削除
            os.remove(uploaded_model.name)

            # 予測
            predictions = classification.predict_model(model, df)
            st.subheader("Predictions Result")
            st.write(predictions)
            with io.BytesIO() as bf:
                predictions.to_csv(bf, index=False)
                href = get_download_link(
                    bf.getvalue(), "predictions.csv", "Download Result(csv)"
                )
            st.markdown(
                f'<div style="text-align: left;">{href}<div>', unsafe_allow_html=True
            )


def predict_regression(uploaded_csv: UploadedFile, uploaded_model: UploadedFile):
    """回帰の予測する処理

    Parameters
    ----------
    uploaded_csv : UploadedFile
        csvファイル
    uploaded_model : UploadedFile
        pklファイル
    """

    if uploaded_csv is not None and uploaded_model is not None:
        st.subheader("Input data")
        df = pd.read_csv(uploaded_csv)
        st.write(df)
        start = st.sidebar.button("Start")
        if start:
            # ファイルを一時保存してモデルを読み込む
            with open(uploaded_model.name, "wb") as f:
                f.write(uploaded_model.getvalue())
            model = regression.load_model(os.path.splitext(uploaded_model.name)[0])

            # 一時保存したファイルを削除
            os.remove(uploaded_model.name)

            # 予測
            predictions = regression.predict_model(model, df)
            st.subheader("Predictions Result")
            st.write(predictions)
            with io.BytesIO() as bf:
                predictions.to_csv(bf, index=False)
                href = get_download_link(
                    bf.getvalue(), "predictions.csv", "Download Result(csv)"
                )
            st.markdown(
                f'<div style="text-align: left;">{href}<div>', unsafe_allow_html=True
            )
