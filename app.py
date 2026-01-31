import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Dự đoán Giá nhà Ames (Pro Version)",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    try:
        # Lưu ý: Hãy đảm bảo tên file ở đây khớp với file .pkl trong máy

        rf_model = joblib.load('house_price_rf.pkl') 
        lr_model = joblib.load('house_price_lr.pkl')
        return rf_model, lr_model
    except Exception as e:
            return None, None

rf_model, lr_model = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040504.png", width=100)
    st.title("Ames Housing AI")
    st.markdown("---")
    st.write("### 👨‍💻👨‍💻👨‍💻👨‍💻👨‍💻")
    st.markdown("**Phát triển bởi: AIO-CONQ016**")
    st.caption("Phiên bản: 1.0 (Stable)")

# --- GIAO DIỆN CHÍNH ---
st.title("🏠 Trợ lý Định giá Bất động sản AI")
st.markdown("---")

# --- DOCS ---
with st.expander("ℹ️ Hướng dẫn: Khi nào nên tin mô hình nào?"):
    st.markdown("""
    ### 1. Linear Regression (Hồi quy tuyến tính)
    * **Phù hợp:** Nhà chung cư, nhà dự án tiêu chuẩn.
    * **Đặc điểm:** Cộng dồn giá trị theo diện tích.
    
    ### 2. Random Forest (Rừng ngẫu nhiên) - KHUYÊN DÙNG
    * **Phù hợp:** Nhà đất thổ cư, nhà phố, biệt thự.
    * **Đặc điểm:** Hiểu được sự tương tác phức tạp (VD: Nhà cũ nhưng vị trí đẹp).
    """)

# --- NHẬP LIỆU ---
col_info, col_qual = st.columns(2)

with col_info:
    st.header("1. Thông số Kỹ thuật")
    gr_liv_area = st.number_input("Diện tích sàn ở (sq ft)", value=1500, step=50)
    total_bsmt_sf = st.number_input("Diện tích hầm (sq ft)", value=1000, step=50)
    garage_area = st.number_input("Diện tích Gara (sq ft)", value=500, step=50)
    
    total_sf = gr_liv_area + total_bsmt_sf
    st.info(f"👉 Tổng diện tích: {total_sf:,} sq ft")
    
    year_built = st.number_input("Năm xây dựng", 1900, 2026, 2010)
    age = 2026 - year_built
    st.caption(f"Tuổi nhà: {age} năm")

with col_qual:
    st.header("2. Đánh giá Chất lượng")
    oa_qual = st.slider("Chất lượng Tổng thể", 1, 10, 6)
    ex_qual = st.slider("Ngoại thất", 1, 5, 3)
    ki_qual = st.slider("Nhà bếp", 1, 5, 3)
    bs_qual = st.slider("Tầng hầm", 1, 5, 3)
    ga_qual = st.slider("Gara", 1, 5, 3)
    
    total_qua = oa_qual + ex_qual + ki_qual + bs_qual + ga_qual
    st.success(f"💎 ĐIỂM CHẤT LƯỢNG: **{total_qua}/30**")

# --- LOGIC GỢI Ý ---
suggested_model = "Random Forest"
suggestion_reason = "Đây là lựa chọn an toàn nhất cho hầu hết các loại nhà đất."

if age > 40:
    suggestion_reason = "⚠️ Nhà cũ (>40 năm). Linear Regression dễ sai số do khấu hao. **Random Forest** chính xác hơn."
elif total_qua > 25:
    suggestion_reason = "🌟 Nhà chất lượng CAO. Giá trị tăng phi tuyến tính. Hãy tin **Random Forest**."
elif total_sf > 4000:
    suggestion_reason = "🏰 Diện tích quá LỚN. Mô hình tuyến tính dễ vỡ trận. **Random Forest** xử lý tốt hơn."

# --- XỬ LÝ DỮ LIỆU ---
input_data = pd.DataFrame({
    'Age': [age],
    'Total_Qua': [total_qua],
    'Total_Qua_Sq': [total_qua ** 2], # <--- THÊM CỘT BÌNH PHƯƠNG
    'TotalSF': [total_sf],
    'Garage Area': [garage_area]
})

# Log Transform (Chỉ Diện tích)
input_data['TotalSF'] = np.log1p(input_data['TotalSF'])
input_data['Garage Area'] = np.log1p(input_data['Garage Area'])

# --- DỰ ĐOÁN ---
st.write("---")
if st.button("🔮 ĐỊNH GIÁ & TƯ VẤN NGAY", type="primary", use_container_width=True):
    if rf_model is None:
        st.error("⚠️ Không tìm thấy file model (.pkl). Hãy kiểm tra lại tên file!")
    else:
        try:
            # Dự đoán
            rf_price = np.expm1(rf_model.predict(input_data)[0])
            lr_price = np.expm1(lr_model.predict(input_data)[0])
            
            st.info(f"💡 **AI Gợi ý:** {suggestion_reason}")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("### 🌲 Random Forest (Khuyên dùng)")
                st.metric("Giá dự báo", f"${rf_price:,.0f}", delta="Độ tin cậy cao")
                st.progress(min(rf_price/1000000, 1.0))
                
            with col_res2:
                st.markdown("### 📈 Linear/Ridge Regression")
                st.metric("Giá tham khảo", f"${lr_price:,.0f}")
                diff = abs(rf_price - lr_price)
                st.caption(f"Chênh lệch: ${diff:,.0f}")
            
            
        except Exception as e:
            st.error(f"Lỗi: {e}")
            
