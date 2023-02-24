import datetime

import streamlit as st
from pycaret import classification


CLASS_PLOT = [
    "auc",
    "threshold",
    "pr",
    "confusion_matrix",
    "error",
    "boundary",
    "learning",
    "vc",
    "feature",
    "manifold",
    "dimension",
]
CLASS_PLOT_NAME = [
    "ROC Curve",  # ROC曲線
    "Threshold",  # 閾値の重要度
    "PR Curve",  # 適合率-再現率
    "Confusion Matrix",  # 混同行列
    "Error",  # どちらに予測したのか
    "Boundary",  # 決定境界
    "Learning Curve",  # 学習曲線
    "Validation Cruve",  # 検証曲線
    "Feature",  # 特徴量の重要度
    "Manifold Learning",  # 多様体学習
    "Dimension",
]


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
    left, center, right = st.columns(3)
    for i, plot in enumerate(CLASS_PLOT):
        if i % 3 == 0:
            with left:
                st.caption(CLASS_PLOT_NAME[i])
                with st.spinner("Wait for it..."):
                    classification.plot_model(
                        model, plot=plot, display_format="streamlit"
                    )
        elif i % 3 == 1:
            with center:
                st.caption(CLASS_PLOT_NAME[i])
                with st.spinner("Wait for it..."):
                    classification.plot_model(
                        model, plot=plot, display_format="streamlit"
                    )
        elif i % 3 == 2:
            with right:
                st.caption(CLASS_PLOT_NAME[i])
                with st.spinner("Wait for it..."):
                    classification.plot_model(
                        model, plot=plot, display_format="streamlit"
                    )
