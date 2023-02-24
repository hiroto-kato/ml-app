"""定数定義"""

TASK_LIST = ["Classification", "Regression", "クラスタリング", "異常検出", "自然言語処理", "アソシエーション分析"]
CLASS_PLOT = [
    ("auc", "ROC Curve"),  # ROC曲線
    ("threshold", "Threshold"),  # 閾値の重要度
    ("pr", "PR Curve"),  # 適合率-再現率
    ("confusion_matrix", "Confusion Matrix"),  # 混同行列
    ("error", "Error"),  # どちらに予測したのか
    ("boundary", "Boundary"),  # 決定境界
    ("learning", "Learning Curve"),  # 学習曲線
    ("vc", "Validation Cruve"),  # 検証曲線
    ("feature", "Feature"),  # 特徴量の重要度
    ("manifold", "Manifold Learning"),  # 多様体学習
    ("dimension", "Dimension"),
]
REGRESSION_PLOT = [
    ("residuals", "Residual"),  # 残差
    ("error", "Prediction Error"),  # 予測誤差
    ("cooks", "Cooks Distance"),  # クック距離
    ("rfe", "Recursive Feat."),
    ("learning", "Learning Curve"),  # 学習曲線
    ("vc", "Validation Curve"),  # 検証曲線
    ("manifold", "Manifold Learning"),  # 多様体学習
    ("feature", "Feature"),  # 特徴量の重要度
]
