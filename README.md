# HR Analytics: Dự đoán Giữ chân Nhân tài (Employee Retention Prediction)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Hardcoded-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

> **Mô tả ngắn:** Dự án xây dựng hệ thống dự đoán khả năng nghỉ việc của nhân sự trong ngành Data Science. Điểm đặc biệt của dự án là việc **tự cài đặt thuật toán Logistic Regression từ con số 0 (from scratch) chỉ sử dụng NumPy**, tích hợp các kỹ thuật nâng cao như Regularization, Class Weighting và Threshold Tuning để giải quyết bài toán mất cân bằng dữ liệu.

---

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Phương pháp & Thuật toán](#phương-pháp--thuật-toán)
4. [Cài đặt & Thiết lập](#cài-đặt--thiết-lập)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Kết quả & Phân tích](#kết-quả--phân-tích)
7. [Cấu trúc Dự án](#cấu-trúc-dự-án)
8. [Thách thức & Giải pháp](#thách-thức--giải-pháp)
9. [Hướng phát triển](#hướng-phát-triển)
10. [Thông tin tác giả](#thông-tin-tác-giả)

---

## 🌟 Giới thiệu

### Mô tả bài toán
Trong nền kinh tế tri thức, "chảy máu chất xám" là cơn ác mộng của mọi doanh nghiệp. Chi phí để tuyển dụng và đào tạo lại một nhân sự Data Scientist là rất lớn. Bài toán đặt ra là: *Làm thế nào để nhận diện sớm những nhân viên có ý định nghỉ việc để HR kịp thời có chính sách giữ chân?*

### Mục tiêu cụ thể
1.  Phân tích các yếu tố ảnh hưởng đến quyết định nghỉ việc (EDA).
2.  Xây dựng mô hình phân loại nhị phân (Binary Classification) để dự báo:
    * `0`: Ổn định (Ở lại).
    * `1`: Rủi ro (Muốn nghỉ việc).
3.  Tối ưu hóa chỉ số **Recall** (để không bỏ sót nhân tài muốn đi) trong bối cảnh dữ liệu bị mất cân bằng nghiêm trọng.

---

## 📊 Dataset

* **Nguồn dữ liệu:** [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
* **Kích thước:** ~19,158 mẫu (bản ghi).
* **Đặc điểm quan trọng:**
    * **Imbalanced Data:** Chỉ có ~25% nhân sự muốn nghỉ việc, 75% ở lại.
    * **Features:** Bao gồm cả định lượng (Training hours, City index) và định tính (Gender, Education, Experience).
    * **Missing Values:** Một số cột như `company_type`, `gender` thiếu dữ liệu lên đến 30%.

| Feature | Mô tả |
| :--- | :--- |
| `city_development_index` | Chỉ số phát triển của thành phố ứng viên sống. |
| `education_level` | Trình độ học vấn (Graduate, Masters, PhD...). |
| `experience` | Số năm kinh nghiệm (0 -> >20 năm). |
| `company_size` | Quy mô công ty hiện tại. |
| `last_new_job` | Khoảng cách giữa lần nhảy việc gần nhất. |
| `training_hours` | Tổng số giờ đào tạo đã hoàn thành. |
| `target` | 0 (Không tìm việc) - 1 (Đang tìm việc). |

---

## ⚙️ Phương pháp & Thuật toán

### 1. Quy trình xử lý dữ liệu (Preprocessing Pipeline)
* **Cleaning:** Điền khuyết (Imputation) chiến lược: dùng Mode cho biến ngẫu nhiên và tạo nhóm 'Unknown' cho biến thiếu có hệ thống.
* **Feature Engineering:**
    * Gom nhóm `city` (Top 10 + Others).
    * Log Transformation cho `training_hours`.
    * Tạo đặc trưng tương tác: `Brain Drain` (Học vấn cao + Vùng kém phát triển).
* **Encoding:**
    * **Ordinal Encoding:** Áp dụng cho biến có thứ tự (`experience`, `education`, `company_size`) để giữ nguyên tính chất lớn bé.
    * **Label/One-Hot Encoding:** Cho các biến định danh.
* **Scaling:** StandardScaler để đưa dữ liệu về phân phối chuẩn ($\mu=0, \sigma=1$).

### 2. Thuật toán: Logistic Regression (NumPy Implementation)
Thay vì dùng thư viện có sẵn, dự án tự cài đặt thuật toán Logistic Regression tối ưu hóa bằng Gradient Descent.

* **Hàm kích hoạt (Sigmoid):**
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Hàm mất mát (Weighted Log Loss với L2 Regularization):**
    Để xử lý mất cân bằng, hàm Loss được điều chỉnh thêm trọng số $w_{class}$:
    $$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \alpha_i [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$
    *(Trong đó $\alpha_i$ là trọng số mẫu, giúp phạt nặng hơn khi đoán sai lớp thiểu số)*.

---

## 🛠 Installation & Setup

1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/yourusername/hr-analytics-numpy.git](https://github.com/yourusername/hr-analytics-numpy.git)
    cd hr-analytics-numpy
    ```

2.  **Tạo môi trường ảo (Khuyên dùng):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Thư viện chính: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost)*

---

## 🚀 Usage

Dự án được chia thành các Notebook theo quy trình chuẩn:

1.  **Khám phá dữ liệu (EDA):**
    * Chạy file `notebooks/01_data_exploration.ipynb`.
    * Xem phân tích Heatmap, Cramér's V correlation và các insight về Brain Drain.

2.  **Tiền xử lý (Preprocessing):**
    * Chạy file `notebooks/02_preprocessing.ipynb`.
    * File này sẽ tạo ra `train_processed.csv` và `test_processed.csv` trong thư mục `data/processed/`.

3.  **Huấn luyện & Đánh giá (Modeling):**
    * Chạy file `notebooks/03_modeling.ipynb`.
    * So sánh kết quả giữa Logistic Regression (Custom NumPy), Random Forest và XGBoost.

---

## 📈 Results

Sau khi tối ưu hóa ngưỡng (Threshold Tuning) và sử dụng Custom Class Weights, mô hình đạt được kết quả khả quan trên tập Test:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom Logistic Reg** | 0.73 | 0.48 | **0.78** | 0.60 | 0.81 |
| **XGBoost (Tuned)** | 0.79 | 0.58 | 0.75 | **0.65** | **0.82** |

**Kết luận:**
* Mô hình đạt **Recall ~78%**, nghĩa là phát hiện được gần 80% nhân viên có ý định nghỉ việc.
* Yếu tố ảnh hưởng lớn nhất: **City Development Index** (Môi trường sống), **Company Size** và **Experience**.
* Yếu tố ít ảnh hưởng: **Gender** và **Training Hours** (Số giờ học không quyết định việc đi hay ở).

---

## 📂 Project Structure