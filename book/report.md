# Báo cáo Phân tích So sánh và Thảo luận Kết quả — Phần (a): Chuỗi thời gian

Báo cáo này tổng hợp số liệu thực nghiệm, kiểm định giả thuyết thống kê và phân tích phê bình cho toàn bộ pipeline tiền xử lý trên chuỗi giá đóng cửa (`Close`) của cổ phiếu **NVIDIA (`NVDA`)** giai đoạn `2000-01-03` – `2026-11-11` ($N = 6,756$ phiên) trong [03_temporal_preprocessing.ipynb](03_temporal_preprocessing.ipynb).

---

## Chương 1: Phân tích Khám phá Chuỗi thời gian (Time Series EDA)

### 1.1. Nhận diện trực quan qua Time Plot
- **Xu hướng (*Trend*):** Cả 4 chuỗi giá (`Close`, `High`, `Low`, `Open`) tăng trưởng phi tuyến dạng tiệm cận hàm mũ, từ mức `0.089206` USD (`2000-01-03`) lên vượt `186` USD cuối năm `2026` (`186.64` USD ngày `2026-11-06`, `173.73` USD ngày `2026-11-11`).
- **Tính mùa vụ (*Seasonality*):** Không bộc lộ chu kỳ lặp lại cố định rõ ràng bằng mắt thường; biến động chịu chi phối bởi chu kỳ bán dẫn/AI và điều kiện vĩ mô.
- **Nhiễu (*Noise*):** Dao động hàng ngày khuếch đại mạnh ở vùng thị giá cao sau năm `2020`; riêng chuỗi `Volume` chịu chi phối bởi nhiễu với nhiều đột biến thanh khoản bất thường.

### 1.2. Cấu trúc tự tương quan (ACF & PACF)
Với $n = 6,756$, ngưỡng trễ tối đa áp dụng công thức:

$$\text{max\_lag} = \left\lfloor \min\left(10 \log_{10} n, \; \frac{n}{5}\right) \right\rfloor = 38$$

Khoảng tin cậy 95% kiểm định nhiễu trắng là $\pm \frac{1.96}{\sqrt{6756}} \approx \pm 0.0238$.
- **ACF:** Suy giảm rất chậm và neo gần mức $1.0$ qua suốt `38` độ trễ, phản ánh ký ức dài hạn và sự hiện diện của xu hướng mạnh.
- **PACF (giải Yule-Walker bằng đệ quy Durbin-Levinson):** Có một đỉnh nhọn duy nhất $\phi_{11} \approx 1.0$ tại $\text{lag} = 1$, sau đó cắt cụt ngay về xấp xỉ $0$ trong vùng tin cậy $95\%$ từ $\text{lag} \ge 2$.
- **Kết luận:** Hình thái này khẳng định chuỗi giá `Close` tuân theo quá trình **Random Walk** bậc 1 ($AR(1)$), hàm ý $Y_t$ bị chi phối chủ yếu bởi $Y_{t-1}$ và nhiễu ngẫu nhiên.

### 1.3. Thống kê trượt đa tỉ lệ (Rolling Mean & Std, $W \in \{7, 30, 90\}$)
- Cửa sổ ngắn $W = 7$ bám sát biến động tức thời; $W = 90$ làm mượt nhiễu tốt nhất nhưng chịu **độ trễ pha** rõ rệt (nằm thấp hơn đường `7D` và `30D` trong pha tăng dài).
- **Rolling Mean** tăng dốc về cuối chuỗi và **Rolling Std** phình to từ gần `0` ở giai đoạn đầu lên biên độ lớn sau `2020`, chứng minh chuỗi gốc vi phạm đồng thời kỳ vọng không đổi và **phương sai đồng nhất (homoscedasticity)**.

---

## Chương 2: Đánh giá Định lượng Kỹ thuật Điền khuyết & Nội suy

### 2.1. Kết quả sai số trên kịch bản khuyết MCAR (`50%`)
Hàm `fault_injection` (`np.random.seed(42)`, tham số mặc định `missing_rate = 0.5`) che ngẫu nhiên không hoàn lại `50.00%` điểm trên `df_close_series`. Năm phương pháp được đo sai số (`MAE`, `RMSE`) và kiểm định phần dư **Ljung-Box** tại $\text{lag} = 38$ ($\alpha = 0.05$):

| Phương pháp | MAE | RMSE | Ljung-Box Full Series ($Q$ / $p$-value) | Ljung-Box trên `missing_indices` ($Q$ / $p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **Forward Fill** | `0.6088` | `2.0776` | `2569.8932` (`0.0`) | `35.5113` (`0.585105`) |
| **Linear Interpolation** ($P_1$) | `0.3890` | `1.2431` | `1220.8845` (`1.724e-231`) | `30.6010` (`0.797629`) |
| **Quadratic Interpolation** ($P_2$) | `0.4733` | `1.5499` | `2363.8880` (`0.0`) | `36.8678` (`0.521703`) |
| **Cubic Interpolation** ($P_3$) | `0.4826` | `1.5852` | `2629.0880` (`0.0`) | `34.8962` (`0.613755`) |
| **Cubic Spline** (từng đoạn bậc 3) | `0.3270`* | `1.1286`* | `1687.6742` (`0.0`) | `44.9954` (`0.202325`) |

> [!NOTE]
> **Nhận xét phê bình về phép đo & hành vi nội suy:**
> 1. **Sai lệch mẫu số ở `Cubic Spline` (\*):** Trong notebook, 4 phương pháp đầu gọi `calc_mae`/`calc_rmse` có truyền `missing_indices`, riêng `Cubic Spline` gọi trên toàn chuỗi (`calc_mae(df_close_series, se_spline)`), khiến `50%` điểm giữ nguyên mang sai số `0` làm giảm giả tạo chỉ số. Nếu quy đổi riêng trên `50%` điểm khuyết, sai số thực chất vào khoảng $\text{MAE} \approx 0.6540$ và $\text{RMSE} \approx 1.5961$. Như vậy **Linear Interpolation** (`MAE = 0.3890`, `RMSE = 1.2431`) mới là phương pháp chính xác nhất.
> 2. **Nghịch lý đa thức bậc cao:** $P_1$ (`0.3890`) vượt trội so với $P_2$ (`0.4733`) và $P_3$ (`0.4826`) do đa thức bậc cao dễ sinh dao động vọt lố (*overshoot* / hiện tượng Runge) khi đi qua nhiễu giật giá của quá trình Random Walk.
> 3. **Forward Fill kém nhất (`0.6088`):** Giả định giữ nguyên giá cũ gây sai số bậc thang lớn trên chuỗi có xu hướng dốc khi khoảng khuyết liên tiếp xuất hiện ở mật độ `50%`.

### 2.2. Thảo luận phê bình về "Ảo giác thống kê" của Ljung-Box
- **Trên toàn bộ chuỗi phần dư (`df_close_series - se_imputed`):** Cả 5 phương pháp đều có $p \approx 0.0 \le 0.05$ (bác bỏ $H_0$), trong đó `Linear` có $Q = 1220.88$ nhỏ nhất. Nguyên nhân là các điểm không khuyết (`= 0`) xen kẽ cùng các cụm nội suy liên tiếp tạo ra cấu trúc phụ thuộc mạnh.
- **Trên tập chỉ mục rời rạc (`missing_indices`):** Cả 5 phương pháp đều cho $p \in [0.2023, 0.7976] > 0.05$ (không bác bỏ $H_0$). Tuy nhiên, vì `missing_indices` rút ngẫu nhiên không giữ trục thời gian đều, cấu trúc tự tương quan bị phá vỡ, gây **ảo giác thống kê** rằng phần dư là nhiễu trắng độc lập.

---

## Chương 3: Kiểm định Tính dừng (Stationarity Tests) & Biến đổi Dữ liệu

Pipeline kết hợp cặp kiểm định đối lập giả thuyết để tránh kết luận phiến diện:
- **ADF** và **PP (Phillips-Perron):** $H_0$: Có nghiệm đơn vị (*không dừng*); bác bỏ $H_0$ khi $p \le 0.05$.
- **KPSS:** $H_0$: Chuỗi *dừng* quanh mức trung bình; bác bỏ $H_0$ khi $p < 0.05$.

| Chuỗi dữ liệu | ADF Stat ($p$-value) | PP Stat ($p$-value) | KPSS Stat ($p$-value) | Kết luận thống nhất ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **Chuỗi gốc (`Close`)** | `1.5361` (`0.9977`) | `1.2053` (`0.9984`) | `5.1545` (`0.0001`) | **Không dừng** |
| **Box-Cox** | `-0.9116` (`0.7841`) | `-1.1496` (`0.6979`) | `11.3222` (`0.0001`) | **Không dừng** |
| **Logarit (`np.log`)** | `0.5368` (`0.9860`) | `0.4107` (`0.9833`) | `10.8874` (`0.0001`) | **Không dừng** |
| **Box-Cox + Sai phân bậc 1** | `-14.0146` (`3.66e-26`) | `-80.4042` (`0.0`) | `0.0341` (`0.9610`) | **Đạt tính dừng** |
| **Log + Sai phân bậc 1** | `-14.1581` (`2.10e-26`) | `-82.8034` (`0.0`) | `0.2109` (`0.2478`) | **Đạt tính dừng** |

**Phân tích tác động:**
- `Box-Cox` và `Log` đơn lẻ chỉ nén thang đo và **ổn định phương sai**, chưa khử được xu hướng nên cả ADF, PP và KPSS đều khẳng định chưa dừng.
- Kết hợp **Sai phân bậc 1** ($\Delta y_t = y_t - y_{t-1}$) triệt tiêu hoàn toàn nghiệm đơn vị và xu hướng, đưa chuỗi về dao động quanh `0`. `Box-Cox + Diff1` tối ưu hơn về mặt ổn định trung bình dài hạn (KPSS $p = 0.9610$), còn `Log + Diff1` (*log-return*) giữ được ý nghĩa tài chính trực tiếp và làm biến mục tiêu lý tưởng cho dự báo.

---

## Chương 4: Phân rã Chuỗi thời gian & Thảo luận Strict RVR

### 4.1. Phân rã Cổ điển (`period = 252`)
- **Additive ($Y_t = T_t + S_t + R_t$):** Từ năm `2020`, phần dư phình rộng tới `[-40, +20]`, bác bỏ giả định nhiễu có phương sai hằng số.
- **Multiplicative ($Y_t = T_t \times S_t \times R_t$):** Phần dư dàn đều quanh trục `1.0` suốt 26 năm, chứng minh bản chất chuỗi giá tài chính mang cấu trúc **nhân tính** (biên độ tỷ lệ theo mức giá).

### 4.2. STL Decomposition (LOESS, `period = 252`, `seasonal = 9`) & Strict RVR
- Chỉ số RVR truyền thống trên chuỗi gốc cho kết quả gây hiểu nhầm:

$$\text{RVR} = \frac{\text{Var}(R_t)}{\text{Var}(Y_t)} = 0.7723\% < 5\%$$

Con số này bị méo mó do mẫu số $\text{Var}(Y_t)$ bị thổi phồng cực đại bởi xu hướng tăng `0.089` $\to$ `186`. Để đánh giá đúng tỷ trọng nhiễu sau khi khử xu hướng, sử dụng **Strict RVR**:

$$\text{Strict RVR} = \frac{\text{Var}(R_t)}{\text{Var}(Y_t - T_t)} = \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)}$$

| Phiên bản chuỗi đầu vào STL | Strict RVR | Hàm ý kỹ thuật |
| :--- | :---: | :--- |
| **Chuỗi `Close` gốc** | `68.3068%` | Hơn 2/3 biến động phi xu hướng là nhiễu ngẫu nhiên |
| **Chuỗi `Box-Cox`** | `68.3951%` | Cấu trúc tỷ lệ nhiễu tương đương chuỗi gốc |
| **Chuỗi `Log`** | `74.0190%` | Nén biên độ vùng giá cao, chuyển nhân tính thành cộng tính chuẩn |

Vì giá cổ phiếu mang cấu trúc nhân tính, chuyển STL lên thang **Log** (`log_trend`, `log_seasonal`, `log_resid`, `log_detrended`, `log_deseasonalized`) biến mô hình tích thành cộng tuyến tính $\ln Y_t = T_t^{(\log)} + S_t^{(\log)} + R_t^{(\log)}$, làm nền cho trích xuất đặc trưng và phát hiện dị thường.

---

## Chương 5: Trích xuất Đặc trưng Thời gian & So sánh 4 Kỹ thuật Phát hiện Dị thường

### 5.1. Pipeline Trích xuất Đặc trưng trên nền Log-STL
1. **Mã hóa chu kỳ tuần hoàn ($\sin/\cos$):** Ngày trong tuần với $T_{\text{tuần}} = 5$ (đã kiểm chứng `0` phiên giao dịch rơi vào T7/CN) và tháng trong năm với $T_{\text{tháng}} = 12$ nhằm bảo toàn tính liên tục khép kín.
2. **Ngày lễ Hoa Kỳ (`holidays.US()`):** Cờ nhị phân `is_holiday` phát hiện `58` phiên giao dịch gắn với ngày lễ Hoa Kỳ.
3. **Thống kê trượt đa tỉ lệ:** Tính `mean`, `std`, `max`, `min`, `skew` trên các cửa sổ $W \in \{7, 30, 90\}$ cho `log_resid`, `log_detrended`, và `log_deseasonalized`.

### 5.2. Đối chiếu 4 Phương pháp Phát hiện Dị thường

| Phương pháp | Chuỗi & Thiết lập đầu vào | Số dị thường | Vùng thời gian tập trung | Đặc trưng & Hạn chế |
| :--- | :--- | :---: | :--- | :--- |
| **1. STL $3\sigma$** | `stl_resid` (thang gốc), $\lvert \text{resid} - \mu \rvert > 3\sigma$ | `248` | `2023-10-19` – `2026-11-11` (`208` điểm từ `2024`) | Thiên lệch nặng về vùng giá tuyệt đối cao |
| **2. Z-score** | `deseasonalized` (còn xu hướng), $\lvert Z \rvert > 3$ | `324` | Cụm dày đặc cuối đuôi (`2025-07-24` trở đi, $Z \in [3.07, 3.21]$) | Bỏ sót sụt giảm `2022` và tăng sốc `2024` vì còn `Trend` và $\mu, \sigma$ toàn cục bị 2000–2020 kéo thấp |
| **3. Isolation Forest** | Rolling stats trên `log_detrended` (`contamination=0.01`, `300` cây): đơn (`W=30`) vs đa tỉ lệ (`W={7,30,90}`, `feat_all`) | `68` (`W=30`) / `67` (`All`) | Giai đoạn `2000–2002` (sụp đổ Dot-com) + cú sụt `2004` (`All`) | Đồng thuận `99.21%` (`6,614/6,667` phiên, khác `53`); bắt tốt biến động tương đối (%) của rìa phân phối |
| **4. Autoencoder 1D** | `StandardScaler(log_detrended)`, `sliding_window_view` (`W=30`, `Conv1d` `1->32->16->32->1`, `100` epochs, ngưỡng quantile `0.99`) | Top `1%` (`> q0.99`) | 4 pha chuyển trạng thái gãy góc: đầu `2000`, chữ V đầu `2001`, gãy `02–03/2002`, hồi phục `2002–2003` | Nhạy với đứt gãy hình thái quỹ đạo nhưng **không đánh dấu đỉnh 2002 (`~0.53`)** |

**Phân tích chuyên sâu sự bổ trợ giữa Isolation Forest và Autoencoder 1D:**
- Tại sao **Isolation Forest** chỉ bắt năm `2000–2002`? Vì trên thang `log_detrended`, độ biến động tương đối (%) thời kỳ vỡ bong bóng Dot-com (rơi >85% từ đỉnh, rung lắc mạnh vùng đáy `0.07–0.12`) lớn hơn nhiều so với biến động % giai đoạn `2025–2026`. Khi nâng từ đơn cửa sổ `30` (bị dải kéo đuôi do 29/30 ngày chồng lấn) lên **đa tỉ lệ `{7, 30, 90}`**, cửa sổ `7` giúp cắt gọn vệt kéo đuôi và bắt chính xác thêm cú đảo chiều chữ V cuối `2000` – đầu `2001` cùng điểm sụt đột ngột năm `2004`, trong khi cửa sổ `90` neo bối cảnh trung hạn.
- Tại sao **Autoencoder 1D** (`L1Loss` hội tụ `0.7330` $\to$ `0.1576`) không bắt đỉnh `2002 (~0.53)`? Hai tầng tích chập `stride=2` hoạt động như **bộ lọc thông thấp**: đỉnh 2002 hình thành tương đối từ từ qua nhiều tháng nên cửa sổ 30 ngày có dạng vòm mượt, đồng thời `log_trend` kịp uốn theo làm `log_detrended` ít gãy khúc, giúp mạng tái tạo với sai số thấp. Ngược lại, Autoencoder phát hiện rất sắc các pha **chuyển chế độ đột ngột** (*regime shifts*), bổ trợ hoàn hảo cho Isolation Forest vốn nhạy với **đuôi phân phối thống kê**.

---

## Chương 6: Đánh giá Định lượng trên Mô hình Dự báo & Kiểm chứng Data Leakage

### 6.1. Thiết kế thực nghiệm dự báo bước `t + 1`
Từ biểu đồ PACF trên chuỗi dừng `log_diff_1` (`n = 6,755`, `max_lag = 38`), các độ trễ vượt ngưỡng tin cậy là `[8, 16, 17, 20, 30, 33]` $\to$ chọn bậc trễ **$p = 8$**. Ma trận đặc trưng `df_full` có kích thước `X: (6665, 32)` (8 trễ `log_ret` + 5 đặc trưng thời gian/lễ + 15 rolling stats trên `log_ret` + 4 đặc trưng STL), biến mục tiêu `y = target_log_ret` ($r_{t+1}$), chia tuần tự theo thời gian **80/20** (`5,332` train / `1,333` test). Giá dự báo khôi phục qua $\hat{P}_{t+1} = \exp(\ln P_t + \hat{r}_{t+1})$.

### 6.2. Bảng đối chiếu Baseline và Audit Rò rỉ dữ liệu (Data Leakage)

| Mô hình (Random Forest / Baseline) | Cấu hình đặc trưng | MAE | RMSE | Directional Accuracy | Đánh giá tính hợp lệ |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Baseline Raw Price** | `8` lag giá thô chưa làm dừng | `75.1553` | `103.1630` | `46.74%` | Mất hoàn toàn khả năng ngoại suy ngoài dải giá train |
| **2. Baseline Naive** | $\hat{P}_{t+1} = P_t$ | `2.0153` | `3.1927` | `N/A` | Điểm chuẩn của quá trình Random Walk |
| **3. Baseline Return-only** | `1` lag `diff` giá đóng cửa | `2.0220` | `3.1937` | `49.51%` | Đã làm dừng nhưng dự báo hướng gần ngẫu nhiên |
| **4. Full Pipeline + STL (Leak)** | Toàn bộ `32` đặc trưng (gồm 4 cột STL) | `1.9946` | `3.1685` | `54.61%` | **Bị look-ahead bias** do STL lọc hai phía `[-t, +t]` |
| **5. Tiền xử lý chuẩn — No-leak** | **`28` đặc trưng thuần quá khứ** | **`2.0139`** | **`3.1905`** | **`52.06%`** | **Mô hình nhân quả hợp lệ cuối cùng** |

### 6.3. Phân tích phê bình về Rò rỉ dữ liệu & Hiệu quả thực chất
1. **Vai trò quyết định của làm dừng dữ liệu:** Chuyển mục tiêu sang `log-return` giúp `RandomForestRegressor` (vốn là mô hình cây không có khả năng ngoại suy vượt ngưỡng lịch sử) giảm **97.32% MAE** (từ `75.1553` xuống `2.0139`), xử lý triệt để bài toán bùng nổ thang giá từ `0.089` lên `> 186` USD.
2. **Audit Data Leakage qua Feature Importance:** Khi kiểm tra Top 10 đặc trưng chi phối `rf_reg` (mô hình 4), cả 4 đặc trưng từ STL toàn chuỗi đều lọt nhóm đầu (`log_trend_diff: 0.0667`, `log_detrended: 0.0609`, `log_resid: 0.0470`, `log_seasonal: 0.0430`). Do LOESS của STL dùng cửa sổ hai phía quanh thời điểm $t$, các biến này âm thầm rò rỉ thông tin tương lai $t+1$. Khi loại bỏ 4 cột này ở mô hình **No-leak** (dùng cấu hình cây nông hơn `max_depth=5, min_samples_leaf=20` để chống overfitting trên chuỗi nhiễu cao), `DirAcc` giảm từ `54.61%` về `52.06%` — xác nhận phần lợi ảo đã bị bóc tách.
3. **Giá trị thực chất & nhận xét phê bình:** Dù không còn rò rỉ, mô hình **No-leak (`28` đặc trưng)** vẫn đạt **MAE (`2.0139`) và RMSE (`3.1905`) thấp nhất** trong các mô hình hợp lệ và nâng **Directional Accuracy lên `52.06%`** (**+2.55 điểm phần trăm** so với Baseline Return-only `49.51%`). Khoảng cách hẹp so với mô hình Naive phản ánh đúng giả thuyết thị trường hiệu quả / Random Walk trên dữ liệu đơn biến. *(Lưu ý phương pháp luận: để so sánh thật sự thuần nhất giữa mô hình 4 và 5, có thể giữ cố định cùng bộ siêu tham số `max_depth`, `min_samples_leaf` thay vì thay đổi đồng thời khi bỏ STL).*

---

## Chương 7: Tổng kết Pipeline & Hướng Mở rộng

1. **Lựa chọn kỹ thuật tối ưu đã kiểm chứng:**
   - *Nội suy:* **Linear Interpolation** (`MAE = 0.3890`) vượt trội đa thức bậc cao trên chuỗi tài chính.
   - *Tính dừng:* **Log + Diff1** (hoặc **Box-Cox + Diff1** cho độ dừng KPSS `0.9610`).
   - *Phân rã:* **Log-STL** kèm đánh giá **Strict RVR**, chỉ dùng làm công cụ **EDA & Anomaly Detection**, **không** đưa STL toàn chuỗi vào bài toán dự báo tương lai.
   - *Dị thường:* Phối hợp **Isolation Forest đa cửa sổ (`7/30/90`)** và **Autoencoder 1D (`Conv1d`)** để bao quát cả bất thường phân phối thống kê lẫn bất thường hình thái chuyển pha.
2. **Định hướng mở rộng:** Bổ sung biến ngoại sinh sẵn có (`Volume`, độ biến động thực hiện, chỉ báo kỹ thuật đa tài sản) và cài đặt phân rã nhân quả (STL mở rộng theo từng cửa sổ trượt chỉ nhìn quá khứ) để cải thiện thêm biên độ dự báo.
