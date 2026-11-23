# Đồ án Tiền xử lý Dữ liệu (Data Preprocessing) — Phần (a): Chuỗi thời gian

> **Môn học:** Khai phá dữ liệu (Data Mining) — Bộ môn Khoa học Máy tính, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM  
> **Phạm vi:** Phân tích thống kê khám phá (EDA), kiểm định giả thuyết thống kê, thiết kế pipeline tiền xử lý có kiểm soát, trích xuất đặc trưng, phát hiện dị thường (gồm Autoencoder 1D) và đánh giá định lượng trên mô hình dự báo (kèm kiểm chứng Data Leakage) đối với **Dữ liệu Chuỗi thời gian (Time Series)**.

---

## 1. Tổng quan Đồ án

Tiền xử lý dữ liệu chuỗi thời gian tài chính đòi hỏi lập luận thống kê chặt chẽ và kiểm soát rò rỉ thông tin theo trục thời gian, vượt xa bước làm sạch đơn thuần. Trong khuôn khổ **Phần (a) — Dữ liệu chuỗi thời gian**, repository này xây dựng một **Jupyter Book** hoàn chỉnh thực nghiệm trên chuỗi giá cổ phiếu **NVIDIA (`NVDA`)** dài hơn 26 năm (`2000-01-03` đến `2026-11-11`):

1. **EDA chuyên sâu:** Time Plot, hàm tự tương quan & tự tương quan riêng phần (**ACF/PACF** tự cài đặt qua Durbin-Levinson, `max_lag = 38`), thống kê trượt đa tỉ lệ ($W \in \{7, 30, 90\}$).
2. **Điền khuyết MCAR (`50%`) & Ljung-Box:** So sánh `Forward Fill`, nội suy đa thức bậc 1–3 (`Linear`, `Quadratic`, `Cubic`) và `Cubic Spline` bằng `MAE`, `RMSE` và kiểm định Ljung-Box ($\text{lag} = 38$), kèm phân tích phê bình hiện tượng Runge và ảo giác thống kê khi kiểm định trên tập chỉ mục rời rạc.
3. **Kiểm định & Biến đổi tính dừng:** Kết hợp bộ ba kiểm định đối nghịch **ADF**, **Phillips-Perron (PP)** và **KPSS** qua 5 trạng thái (`Close`, `Box-Cox`, `Log`, `Box-Cox + Diff1`, `Log + Diff1`).
4. **Phân rã chuỗi & Strict RVR:** Đối chiếu **Additive**, **Multiplicative** và **STL (LOESS)** (`period = 252`, `seasonal = 9`); chứng minh sai lệch mẫu số của RVR truyền thống (`0.7723%`) bằng **Strict RVR** (`68.31%` / `68.40%` / `74.02%`) và chuẩn hóa phân rã trên thang **Log**.
5. **Trích xuất đặc trưng & Phát hiện dị thường:** Mã hóa chu kỳ ($\sin/\cos$ với $T_{\text{tuần}} = 5$, $T_{\text{tháng}} = 12$), lịch ngày lễ Hoa Kỳ (`holidays.US()`, `58` phiên), thống kê trượt (`mean`, `std`, `max`, `min`, `skew`) trên Log-STL; đối chiếu **4 phương pháp phát hiện dị thường**: STL $3\sigma$ (`248` điểm), Z-score (`324` điểm), **Isolation Forest** đơn (`68` điểm) vs đa tỉ lệ (`67` điểm, đồng thuận `99.21%`) và **Autoencoder 1D** (`Conv1d`, PyTorch, `100` epochs, loss `0.7330` $\to$ `0.1576`).
6. **Đánh giá định lượng & Kiểm chứng Data Leakage:** Dùng `RandomForestRegressor` (PACF `p = 8`, `32` đặc trưng, split `80/20` theo thời gian) đối chiếu với 3 baseline (`Raw Price`, `Naive`, `Return-only`). Phát hiện *look-ahead bias* từ 4 đặc trưng STL toàn chuỗi và thiết lập mô hình **No-leak** (`28` đặc trưng nhân quả): giảm **97.32% MAE** so với Raw Price (`75.1553` $\to$ `2.0139`) và đạt **Directional Accuracy = 52.06%**.

---

## 2. Thông tin Sinh viên

| Thông tin | Chi tiết |
| :--- | :--- |
| **Họ và tên** | `[Họ và tên sinh viên]` |
| **MSSV** | `[Mã số sinh viên]` |
| **Lớp / Khóa** | `[Lớp / Khóa học]` |
| **Email liên hệ** | `[Email sinh viên]` |
| **Jupyter Book (GitHub Pages)** | `[https://<username>.github.io/<repo>/]` |

> **Lưu ý:** Vui lòng cập nhật các trường `[...]` bằng thông tin cá nhân và URL GitHub Pages công khai trước khi nộp link lên Moodle.

---

## 3. Mô tả Tập dữ liệu & Liên kết Tài nguyên ngoài

- **Tập dữ liệu:** Lịch sử giá giao dịch theo ngày của cổ phiếu **NVIDIA Corporation (`NVDA`)**
- **Giai đoạn:** `2000-01-03` đến `2026-11-11` (`6,756` phiên giao dịch ngày làm việc, không có giao dịch Thứ 7 / Chủ nhật)
- **Kích thước & Chất lượng:** `6,756` dòng × `5` cột (`~316.7 KB`), `0` giá trị thiếu tự nhiên (`6,756 non-null`), giá `Close` tăng từ `0.089206` USD (`2000-01-03`) lên vượt `186` USD (`186.64` USD ngày `2026-11-06`, `173.73` USD ngày `2026-11-11`).
- **Chiến lược lưu trữ dữ liệu:** Không đẩy dữ liệu thô dung lượng lớn vào git nếu vượt giới hạn; mã tải dữ liệu tự động qua `yfinance` (`yf.download("NVDA", start="2000-01-01", end="2026-11-12")`) được tích hợp ngay đầu notebook [book/03_temporal_preprocessing.ipynb](book/03_temporal_preprocessing.ipynb) và xuất ra `data/raw/timeseries_a/nvda_raw.csv`.

| Thuộc tính | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `Date` (Index) | `DatetimeIndex` | Ngày giao dịch thị trường chứng khoán Mỹ |
| `Close` | `float64` | Giá đóng cửa điều chỉnh (chuỗi phân tích chính) |
| `High` | `float64` | Giá cao nhất trong phiên |
| `Low` | `float64` | Giá thấp nhất trong phiên |
| `Open` | `float64` | Giá mở cửa trong phiên |
| `Volume` | `int64` | Khối lượng cổ phiếu khớp lệnh |

### Liên kết tài nguyên ngoài
- **Tập dữ liệu tham chiếu công khai (Kaggle):** [Stock Market Data — paultimothymooney](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data)
- **Nguồn tải trực tiếp trong notebook:** Yahoo Finance API thông qua thư viện `yfinance==1.8.0` (`NVDA`)
- **Đặc tả yêu cầu đồ án (HCMUS CS-TA):** [Tiền xử lý dữ liệu — Data Mining Lab Book](https://hcmus-cs-ta.github.io/lab-book/data-mining/projects/preprocessing/)

---

## 4. Cấu trúc Repository

```text
data-mining/
├── README.md                                  # Tổng quan, thông tin SV, mô tả dữ liệu, hướng dẫn cài đặt & build
├── requirements.txt                           # Thư viện Python và phiên bản môi trường
├── book/                                      # Mã nguồn cuốn Jupyter Book
│   ├── index.md                               # Trang mở đầu giới thiệu quy trình & pipeline
│   ├── 03_temporal_preprocessing.ipynb        # Notebook EDA, tiền xử lý, Autoencoder 1D & đánh giá RF
│   └── report.md                              # Báo cáo phân tích định lượng & thảo luận phê bình
└── data/
    ├── raw/
    │   └── timeseries_a/
    │       └── nvda_raw.csv                   # Dữ liệu thô OHLCV NVDA (6,756 × 5)
    └── processed/
        └── timeseries_a/                      # 33 file CSV kết quả từng giai đoạn
            ├── se_ff.csv, se_p1.csv, se_p2.csv, se_p3.csv, se_spline.csv
            ├── se_box_cox.csv, se_log.csv, box_cox_diff_1.csv, log_diff_1.csv
            ├── stl_trend.csv, stl_seasional.csv, stl_resid.csv
            ├── detrended.csv, deseasonalized.csv
            ├── df_detrend_day_extract.csv, df_deseasonal_day_extract.csv
            ├── df_detrend_month_extract.csv, df_deseasonal_month_extract.csv
            ├── df_residual_extract_{7,30,90}.csv
            ├── df_detrend_extract_{7,30,90}.csv
            ├── df_deseasonal_extract_{7,30,90}.csv
            ├── anomolies.csv, anomaly_z_score.csv
            ├── df_iforest_size_30.csv, df_iforest_all.csv, df_compare.csv
            └── feat_all.csv
```

---

## 5. Hướng dẫn Cài đặt Môi trường, Chạy Notebook & Build Jupyter Book

### Bước 1: Tạo môi trường ảo và cài đặt phụ thuộc (`Python >= 3.10`)
```bash
git clone <repository-url>
cd data-mining

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

> **Thư viện chính sử dụng:** `numpy==2.5.4`, `pandas==3.1.1`, `scipy==1.18.1`, `statsmodels==0.15.0`, `arch==9.0.0`, `scikit-learn==1.9.1`, `torch==2.15.0`, `holidays==0.108`, `yfinance==1.8.0`, `matplotlib==3.11.2`.

### Bước 2: Chạy sạch Notebook từ đầu đến cuối
1. Mở [book/03_temporal_preprocessing.ipynb](book/03_temporal_preprocessing.ipynb) trong JupyterLab / VS Code.
2. Chọn kernel `.venv`, đảm bảo có kết nối Internet để `yfinance` tải dữ liệu `NVDA`.
3. Chạy **Run All** từ cell đầu đến cell cuối để tái tạo toàn bộ biểu đồ, kiểm định, kết quả huấn luyện và tự động xuất `data/raw/timeseries_a/nvda_raw.csv` cùng 33 file CSV trong `data/processed/timeseries_a/`.

### Bước 3: Cài đặt công cụ dựng sách và xuất bản Jupyter Book
Cần đảm bảo đã có cấu hình mục lục `book/_toc.yml` và `book/_config.yml` trong thư mục `book/` trước khi build:

```bash
pip install jupyter-book ghp-import

# Build phiên bản HTML tĩnh của cuốn book (không chạy lại hoặc chạy trực tiếp tùy cấu hình)
jupyter-book build book/

# Xuất bản lên nhánh gh-pages của repository GitHub cá nhân
ghp-import -n -p -f book/_build/html
```
