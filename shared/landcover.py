"""
토지피복 비율 계산 함수 (담당: B, D도 지도 격자에 그대로 재사용)
- ESA WorldCover 10m 지도(GeoTIFF, 위경도 좌표계)에서
  한 지점 주변 반경 r(m) 원 안의 녹지·불투수면·수공간 비율을 계산

WorldCover 클래스 코드:
  10 나무, 20 관목, 30 초지, 40 농경지, 50 건물(시가지), 60 나지,
  80 영구 수역, 90 습지  (그 외 70, 95, 100은 서울에 거의 없음)
"""
import numpy as np
import rasterio
from rasterio.windows import from_bounds

GREEN = [10, 20, 30, 40, 90]
BUILT = [50]
WATER = [80]


def landcover_ratio(src, lat, lon, radius_m=300):
    """
    src : rasterio.open(...)으로 연 WorldCover 지도
    반환: {"green": 0.3, "impervious": 0.6, "water": 0.1}
    """
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * np.cos(np.radians(lat)))
    win = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform)
    arr = src.read(1, window=win, boundless=True, fill_value=0)
    if arr.size == 0:
        return {"green": np.nan, "impervious": np.nan, "water": np.nan}

    # 정사각형 창 안에서 '원' 안쪽 픽셀만 남기기
    t = src.window_transform(win)
    rows, cols = np.indices(arr.shape)
    px_lon = t.c + (cols + 0.5) * t.a
    px_lat = t.f + (rows + 0.5) * t.e
    dist = np.sqrt(((px_lat - lat) * 111_320) ** 2 +
                   ((px_lon - lon) * 111_320 * np.cos(np.radians(lat))) ** 2)
    vals = arr[(dist <= radius_m) & (arr > 0)]
    if vals.size == 0:
        return {"green": np.nan, "impervious": np.nan, "water": np.nan}
    return {
        "green": float(np.isin(vals, GREEN).mean()),
        "impervious": float(np.isin(vals, BUILT).mean()),
        "water": float(np.isin(vals, WATER).mean()),
    }


def add_landcover_columns(df, tif_path, radius_m=300):
    """df(lat, lon 열 필요)에 green_300, impervious_300, water_300 열을 추가해서 반환"""
    out = df.copy()
    with rasterio.open(tif_path) as src:
        res = [landcover_ratio(src, la, lo, radius_m) for la, lo in zip(out["lat"], out["lon"])]
    for k in ["green", "impervious", "water"]:
        out[f"{k}_{radius_m}"] = [r[k] for r in res]
    return out
