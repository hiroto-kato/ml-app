import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
from pycaret import classification
from pycaret.datasets import get_data


TASK_LIST = ["分類", "回帰", "クラスタリング", "異常検出", "自然言語処理", "アソシエーション分析"]


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
        st.header("読み込みデータの表示")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # ファイルは後で
        else:
            df = get_data("employee")
        st.write(df)

        # 学習データとテストデータに分割
        df_train = df.sample(frac=0.95, random_state=1).reset_index(drop=True)
        df_valid = df.drop(df_train.index).reset_index(drop=True)

        # 目的変数の選択
        target = st.sidebar.selectbox("目的変数を選択", list(df.columns))
        df_drop_target = df.drop(target, axis=1)  # 目的変数を除いた特徴量

        # 使用しない特徴量を選択する
        ignore_features = st.sidebar.multiselect(
            "使用しない特徴量を選択（未選択可）", list(df_drop_target)
        )

        # 前処理
        exp = classification.setup(
            df_train, target=target, ignore_features=ignore_features, silent=True
        )
        st.write(exp)
        df = classification.get_config("logging_param")
        compare_models = classification.compare_models()
        st.write(compare_models)


def main():
    # faviconの設定など
    st.set_page_config(page_icon="", page_title="PyCaret App")
    st.title("PyCaret ML Application")
    st.write("PyCaretで機械学習")
    st.write("")

    # サイドバー
    task = st.sidebar.selectbox("機械学習タスクを選択", TASK_LIST)
    if task == "分類":
        uploaded_file = st.sidebar.file_uploader("ファイルをアップロード", type="csv")
        is_sample_data = st.sidebar.checkbox(
            "サンプルデータを使用", value=False
        )  # uploaded_fileある時はFalseにしたい

        # メイン画面
        main_classification(uploaded_file, is_sample_data)

    elif task == "回帰":
        uploaded_file = st.sidebar.file_uploader("ファイルをアップロード", type="csv")
    elif task == "クラスタリング":
        st.write("Coming Soon")
    elif task == "異常検出":
        st.write("Coming Soon")
    elif task == "自然言語処理":
        st.write("Coming Soon")
    elif task == "アソシエーション分析":
        st.write("Coming Soon")


if __name__ == "__main__":
    main()
