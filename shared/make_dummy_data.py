"""
가짜 데이터 생성기 (진짜 데이터가 나오기 전에 모델 코드를 미리 짜기 위한 용도)
- 열 이름·형식은 진짜 파일과 똑같이 맞춘다 → 나중에 파일 경로만 바꾸면 됨
- ⚠️ 값은 전부 무작위로 만든 가짜. 결과 해석에 쓰면 안 됨!

실행: python make_dummy_data.py  (또는 노트북에서 import 후 make_all(BASE))
"""
import os
import numpy as np
import pandas as pd

GU = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구",
      "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
      "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]


def make_all(base_dir, n=1100, seed=42):
    rng = np.random.default_rng(seed)
    gu = rng.choice(GU, n)
    # 구마다 대충 중심점을 정하고 그 주변에 센서를 뿌림 (가짜)
    centers = {g: (rng.uniform(37.45, 37.68), rng.uniform(126.82, 127.15)) for g in GU}
    lat = np.array([centers[g][0] for g in gu]) + rng.normal(0, 0.01, n)
    lon = np.array([centers[g][1] for g in gu]) + rng.normal(0, 0.01, n)

    green = rng.beta(2, 4, n)
    water = rng.beta(1, 15, n)
    imperv = np.clip(1 - green - water - rng.uniform(0, 0.1, n), 0, 1)
    dist_han = rng.uniform(0, 8000, n)

    # 가짜 규칙: 불투수면↑ 더움, 녹지↑ 시원, 한강 가까우면 시원 + 잡음
    uhi_night = 1.5 * imperv - 1.0 * green - 0.5 * np.exp(-dist_han / 1000) + rng.normal(0, 0.3, n)
    uhi_day = 0.8 * imperv - 0.6 * np.sqrt(green) + rng.normal(0, 0.5, n)

    ids = [f"S{i:04d}" for i in range(n)]
    labels = pd.DataFrame({"sensor_id": ids, "gu": gu, "lat": lat, "lon": lon,
                           "uhi_day": uhi_day.round(3), "uhi_night": uhi_night.round(3)})
    feats = pd.DataFrame({"sensor_id": ids, "gu": gu, "lat": lat, "lon": lon,
                          "green_300": green.round(3), "impervious_300": imperv.round(3),
                          "water_300": water.round(3), "dist_han_m": dist_han.round(0)})
    train = labels.merge(feats.drop(columns=["gu", "lat", "lon"]), on="sensor_id")

    out = os.path.join(base_dir, "data_processed")
    os.makedirs(out, exist_ok=True)
    labels.to_csv(os.path.join(out, "labels_dummy.csv"), index=False)
    feats.to_csv(os.path.join(out, "features_dummy.csv"), index=False)
    train.to_csv(os.path.join(out, "train_dummy.csv"), index=False)
    return out


if __name__ == "__main__":
    print(make_all(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
