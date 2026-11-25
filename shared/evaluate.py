"""
공통 평가 함수 (담당: C)  — 강의 6주차 '팀 프로젝트 신뢰성 프로토콜' + 5주차 '공정한 모델 비교'를 따름

규칙 (강의안 그대로):
- 분할은 배포 상황을 따라간다: 새 '동네(구)'를 예측하는 문제 → 자치구 단위로 나눈다 (6주차 Spatial/Group split)
- 모든 모델은 같은 폴드로 평가한다 (5주차 Same folds)
- 전처리(스케일링 등)는 반드시 Pipeline 안에 넣어 학습 폴드에서만 fit (6주차 Leakage)
- 폴드별 평균 ± 표준편차를 보고한다 (5주차 Report spread)
- 시도한 모든 모델·설정을 기록한다 (6주차 Lock the test set, keep a log)
- 테스트 점수에는 95% 신뢰구간을 붙인다 (6주차 Report an interval)

사용 예:
    from evaluate import cv_by_district, save_cv_result
    overall, per_gu, oof = cv_by_district(model, X, y, groups=df["gu"])
    save_cv_result("model3_RF", "uhi_night", overall, BASE, note="n_estimators=300")
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42


def metrics(y_true, y_pred):
    """RMSE, MAE(℃ 단위), R² 계산"""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cv_by_district(model, X, y, groups, n_splits=None):
    """
    자치구 단위 교차검증.
    - n_splits=None : 구 하나씩 빼고 검증 (Leave-One-District-Out)
    - n_splits=5    : 구를 5묶음으로 나눠 검증 (더 빠름)
    ⚠️ model에 스케일러 등 전처리가 있으면 make_pipeline(...)으로 묶어서 넣기

    반환:
      overall : 전체 성능 dict (RMSE, MAE, R2 + 폴드별 RMSE 평균·표준편차)
      per_gu  : 검증 때 뺀 구별 성능 DataFrame
      oof     : 센서마다 '자기 구가 빠졌을 때'의 예측값 → 실패 사례·잔차 분석용
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    splitter = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    rows = []
    for tr, va in splitter.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        oof[va] = pred
        r = metrics(y.iloc[va], pred)
        r.update({"held_out_gu": ", ".join(sorted(groups.iloc[va].unique())), "n_sensors": len(va)})
        rows.append(r)

    per_gu = pd.DataFrame(rows)[["held_out_gu", "n_sensors", "RMSE", "MAE", "R2"]]
    overall = metrics(y, oof)
    overall["RMSE_fold_mean"] = float(per_gu["RMSE"].mean())
    overall["RMSE_fold_sd"] = float(per_gu["RMSE"].std())
    return overall, per_gu, oof


def nested_cv_by_district(search, X, y, groups):
    """
    튜닝까지 포함한 정직한 점수 (6주차: Nested CV)
    - 바깥: 구 하나씩 빼고 채점 / 안쪽: 남은 구로 GroupKFold 튜닝
    - search 예: GridSearchCV(XGBRegressor(...), param_grid, cv=GroupKFold(5),
                               scoring="neg_root_mean_squared_error")
    반환: overall, per_gu, oof, 폴드마다 고른 설정 목록
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    rows, chosen = [], []
    for tr, va in LeaveOneGroupOut().split(X, y, groups):
        s = clone(search)
        s.fit(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])
        pred = s.predict(X.iloc[va])
        oof[va] = pred
        chosen.append(s.best_params_)
        r = metrics(y.iloc[va], pred)
        r.update({"held_out_gu": groups.iloc[va].iloc[0], "n_sensors": len(va)})
        rows.append(r)
    per_gu = pd.DataFrame(rows)[["held_out_gu", "n_sensors", "RMSE", "MAE", "R2"]]
    overall = metrics(y, oof)
    overall["RMSE_fold_mean"] = float(per_gu["RMSE"].mean())
    overall["RMSE_fold_sd"] = float(per_gu["RMSE"].std())
    return overall, per_gu, oof, chosen


def save_cv_result(model_name, target, overall, base_dir, note=""):
    """
    교차검증 결과를 results/cv_scores.csv에 '추가'만 함 (덮어쓰지 않음 = 시도 기록 로그)
    note에 바꾼 설정을 적기. 비교표는 latest_scores()로 모델별 마지막 줄만 뽑아 씀
    """
    path = os.path.join(base_dir, "results", "cv_scores.csv")
    row = pd.DataFrame([{"time": datetime.now().strftime("%m-%d %H:%M"), "model": model_name,
                         "target": target, **{k: round(v, 4) for k, v in overall.items()}, "note": note}])
    if os.path.exists(path):
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)
    return path


def latest_scores(base_dir):
    """모델·타깃별 마지막 기록만 → 발표용 비교표 (몇 번 시도했는지 n_tries도 같이 보고)"""
    log = pd.read_csv(os.path.join(base_dir, "results", "cv_scores.csv"))
    n = log.groupby(["model", "target"]).size().rename("n_tries")
    return log.groupby(["model", "target"]).tail(1).set_index(["model", "target"]).join(n).reset_index()


def worst_sensors(df, y, oof, k=10):
    """오차가 가장 큰 센서 k개 (실패 사례 후보)"""
    out = df.copy().reset_index(drop=True)
    out["true"] = pd.Series(y).reset_index(drop=True)
    out["pred"] = oof
    out["error"] = out["pred"] - out["true"]
    out["abs_error"] = out["error"].abs()
    return out.sort_values("abs_error", ascending=False).head(k)


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=RANDOM_STATE):
    """테스트 RMSE의 95% 신뢰구간 (6주차: 테스트 센서를 복원추출 2,000번)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), (n_boot, len(y_true)))
    rmses = np.sqrt(((y_true[idx] - y_pred[idx]) ** 2).mean(axis=1))
    return float(np.sqrt(((y_true - y_pred) ** 2).mean())), np.percentile(rmses, [2.5, 97.5])


def paired_bootstrap(y_true, pred_a, pred_b, n_boot=2000, seed=RANDOM_STATE):
    """
    같은 테스트 센서에서 두 모델 비교 (6주차: paired bootstrap)
    반환: RMSE 차이(A-B)의 95% 구간. 0을 포함하면 '통계적으로 구분 안 됨'
    """
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    d = np.sqrt(((y[idx] - a[idx]) ** 2).mean(1)) - np.sqrt(((y[idx] - b[idx]) ** 2).mean(1))
    return np.percentile(d, [2.5, 97.5])


def perm_importance(fitted_model, X, y, seed=RANDOM_STATE):
    """순열 중요도 (5주차: 트리 기본 중요도(MDI)보다 믿을 만함). RMSE가 얼마나 나빠지는지(℃)"""
    r = permutation_importance(fitted_model, X, y, n_repeats=20, random_state=seed,
                               scoring="neg_root_mean_squared_error")
    return pd.DataFrame({"feature": X.columns, "RMSE_increase": r.importances_mean,
                         "sd": r.importances_std}).sort_values("RMSE_increase", ascending=False)


def training_range(X):
    """학습 데이터의 변수별 범위 (6주차: domain of applicability 보고용)"""
    return pd.DataFrame({"min": X.min(), "max": X.max()})


def outside_range(X_new, X_train):
    """새 지점(지도 격자 등)이 학습 범위 밖인지 표시 → 트리는 외삽 못 함(5·6주차)"""
    lo, hi = X_train.min(), X_train.max()
    return ((X_new < lo) | (X_new > hi)).any(axis=1)
