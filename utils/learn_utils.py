import os
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
from pycaret.datasets import get_data

from utils.general import *
from utils.lib import *


def main_classification(uploaded_file: UploadedFile, is_sample_data: bool):
    """分類のメイン画面

    Parameters
    ----------
    uploaded_file : UploadedFile
        アップロードファイル
    is_sample_data : boolean
        サンプルデータを使用するかどうか
    """
    if (uploaded_file is not None) or (is_sample_data and uploaded_file is None):
        st.subheader("Input data")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = get_data("employee")
        st.write(df)

        # 目的変数の選択
        target = st.sidebar.selectbox("Select target variable", list(df.columns))
        df_drop_target = df.drop(target, axis=1)  # 目的変数を除いた特徴量

        # 使用しない特徴量を選択する
        ignore_features = st.sidebar.multiselect(
            "Select unused features", list(df_drop_target)
        )
        is_analysis = st.sidebar.checkbox("Analyze (require time)")
        start = st.sidebar.button("Start")
        if start:
            # 分類処理
            model, model_name = process_classification(df, target, ignore_features)

            if is_analysis:
                # 解析したデータをグラフで表示
                plot_classification(model)

            # モデルの保存
            with open(model_name, "rb") as f:
                href = get_download_link(f.read(), model_name, "Download Model")
            st.markdown(
                f'<div style="text-align: left;">{href}<div>', unsafe_allow_html=True
            )

            os.remove(model_name)


def main_regression(uploaded_file: UploadedFile, is_sample_data: bool):
    """回帰のメイン画面

    Parameters
    ----------
    uploaded_file : UploadedFile
        アップロードファイル
    is_sample_data : boolean
        サンプルデータを使用するかどうか
    """
    if (uploaded_file is not None) or (is_sample_data and uploaded_file is None):
        st.subheader("Input data")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = get_data("boston")
        st.write(df)

        # 目的変数の選択
        target = st.sidebar.selectbox("Select target variable", list(df.columns))
        df_drop_target = df.drop(target, axis=1)  # 目的変数を除いた特徴量

        # 使用しない特徴量を選択する
        ignore_features = st.sidebar.multiselect(
            "Select unused features", list(df_drop_target)
        )
        is_analysis = st.sidebar.checkbox("Analyze (require time)")
        start = st.sidebar.button("Start")
        if start:
            # 分類処理
            model, model_name = process_regression(df, target, ignore_features)

            if is_analysis:
                # 解析したデータをグラフで表示
                plot_regression(model)

            # モデルの保存
            with open(model_name, "rb") as f:
                href = get_download_link(f.read(), model_name, "Download Model")
            st.markdown(
                f'<div style="text-align: left;">{href}<div>', unsafe_allow_html=True
            )

            os.remove(model_name)
