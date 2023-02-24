import datetime

import streamlit as st
from pycaret import classification, regression

from core.constants import *


def process_classification(
    df,
    target,
    ignore_features,
):
    """分類のメイン処理
    セットアップ、モデル比較、パラメータチューニング、アンサンブル学習、予測

    Parameters
    ----------
    df : DataFrame
        全データ
    target : Any
        目的変数
    ignore_features : Any
        使用しない特徴量

    Returns
    -------
    model
        学習モデル
    """
    # セットアップ。全データの70%を学習データとする
    exp = classification.setup(
        df, target=target, ignore_features=ignore_features, silent=True
    )

    best_model = None
    tuned_model = None
    model = None
    with st.spinner("Processing compare models..."):
        # モデルの比較
        best_model = classification.compare_models(errors="raise")
        st.subheader("Compare Models")
        st.write(classification.pull())  # 比較結果の表示

    with st.spinner("Processing hyperparameter tuning..."):
        # ハイパーパラメータのチューニング
        tuned_model = classification.tune_model(best_model, n_iter=10)

    with st.spinner("Proceesing ensemble model..."):
        pass
        # アンサンブルモデルの作成
        # model = classification.ensemble_model(tuned_model)
        # チューニング後のモデルを表示。indexの型を変換しないと表示がうまくできない
        # st.write(pull().rename(index=str))

    # 予測
    st.subheader("Predictions")
    predictions = classification.predict_model(tuned_model)
    st.write(predictions)

    # モデルの保存
    saved_model, model_name = classification.save_model(
        tuned_model,
        "classification_saved_model_" + datetime.date.today().strftime("%Y%m%d"),
    )

    return tuned_model, model_name


def plot_classification(model):
    """データ解析用のグラフを並べて表示する（分類）

    Parameters
    ----------
    model :
        学習後のモデル（分類）
    """
    st.subheader("Analysis")
    left, center = st.columns(2, gap="large")
    for i, plot in enumerate(CLASS_PLOT):
        if i % 2 == 0:
            with left:
                _plot_model_class(plot, model)
        elif i % 2 == 1:
            with center:
                _plot_model_class(plot, model)


def _plot_model_class(plot, model):
    """plot_modelの実行

    Parameters
    ----------
    plot : (plot_name, desiplay_plot_name)
        (プロット名、表示名)
    model : _type_
        学習済みモデル
    """
    st.caption(plot[1])
    with st.spinner("Wait for it..."):
        classification.plot_model(model, plot=plot[0], display_format="streamlit")


def process_regression(
    df,
    target,
    ignore_features,
):
    """回帰のメイン処理
    セットアップ、モデル比較、パラメータチューニング、アンサンブル学習(なし)、予測

    Parameters
    ----------
    df : DataFrame
        全データ
    target : Any
        目的変数
    ignore_features : Any
        使用しない特徴量

    Returns
    -------
    model
        学習モデル
    """
    pass

    # セットアップ。全データの70%を学習データとする
    exp = regression.setup(
        df, target=target, ignore_features=ignore_features, silent=True
    )

    best_model = None
    tuned_model = None
    model = None
    with st.spinner("Processing compare models..."):
        # モデルの比較
        best_model = regression.compare_models(errors="raise")
        st.subheader("Compare Models")
        st.write(regression.pull())  # 比較結果の表示

    with st.spinner("Processing hyperparameter tuning..."):
        # ハイパーパラメータのチューニング
        tuned_model = regression.tune_model(best_model, n_iter=10)

    with st.spinner("Proceesing ensemble model..."):
        pass
        # アンサンブルモデルの作成
        # model = regression.ensemble_model(tuned_model)
        # チューニング後のモデルを表示。indexの型を変換しないと表示がうまくできない
        # st.write(pull().rename(index=str))

    # 予測
    st.subheader("Predictions")
    predictions = regression.predict_model(tuned_model)
    st.write(predictions)

    # モデルの保存
    saved_model, model_name = regression.save_model(
        tuned_model,
        "regression_saved_model_" + datetime.date.today().strftime("%Y%m%d"),
    )

    return tuned_model, model_name


def plot_regression(model):
    """データ解析用のグラフを並べて表示する（回帰）

    Parameters
    ----------
    model :
        学習後のモデル（回帰）
    """
    st.subheader("Analysis")
    left, center, right = st.columns(2, gap="large")
    for i, plot in enumerate(REGRESSION_PLOT):
        if i % 2 == 0:
            with left:
                _plot_model_regression(plot, model)
        elif i % 2 == 1:
            with center:
                _plot_model_regression(plot, model)


def _plot_model_regression(plot, model):
    """plot_modelの実行

    Parameters
    ----------
    plot : (plot_name, desiplay_plot_name)
        (プロット名、表示名)
    model : _type_
        学習済みモデル
    """
    st.caption(plot[1])
    with st.spinner("Wait for it..."):
        regression.plot_model(model, plot=plot[0], display_format="streamlit")
