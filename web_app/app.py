"""
SPTS NIDS Simulation - Streamlit Dashboard (Tier 1 - Presentation Layer).

Dashboard mô phỏng tấn công đối nghịch (Adversarial Attack Traffic - AAT) lên hệ thống
phát hiện xâm nhập mạng (NIDS). Cho phép chọn model, thuật toán tấn công, điều chỉnh
siêu tham số và xem kết quả Evasion Rate cùng phân phối 4 đặc trưng Level-1.
"""

import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import io
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.utils import STORAGE_DIR
from web_app.attack_service import (
    get_available_models,
    build_paths_from_models,
    generate_aat_from_data,
    LEVEL1_FEATURES,
)
from database.db_manager import init_db, log_simulation, get_history

# Session state for last run (Compare / ExplainAI)
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

# Page config
st.set_page_config(
    page_title="SPTS NIDS Simulation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize DB
init_db()

# Sidebar - Config
st.sidebar.header("⚙️ Cấu hình")

with st.sidebar.expander("📖 Hướng dẫn nhanh"):
    st.markdown("""
    **Luồng sử dụng:**
    1. Chọn **Target Model** (NIDS cần đánh giá) và **Surrogate Model** (dùng để tạo tấn công)
    2. Chọn thuật toán và điều chỉnh tham số
    3. Bấm **Generate AAT** để tạo luồng tấn công đối nghịch
    4. Xem **Evasion Rate** (tỷ lệ bypass NIDS) và biểu đồ phân phối đặc trưng

    **Lưu ý:** Target và Surrogate nên cùng dataset (vd: `ton_mlp_t` + `ton_mlp_s`).
    """)

target_models, surrogate_models = get_available_models()

if not target_models or not surrogate_models:
    st.sidebar.warning(
        f"Không tìm thấy model trong `{os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models')}`. "
        "Vui lòng train model trước."
    )
    target_model = surrogate_model = None
else:
    target_model = st.sidebar.selectbox(
        "Target Model",
        options=target_models,
        index=0,
        help="Mô hình NIDS cần đánh giá độ robust. Kết quả AAT sẽ được đưa qua model này để tính Evasion Rate.",
    )
    surrogate_model = st.sidebar.selectbox(
        "Surrogate Model",
        options=surrogate_models,
        index=0,
        help="Mô hình dùng để tạo nhiễu tấn công. Thuật toán sẽ tối ưu dựa trên gradient của model này.",
    )
    # Hiển thị nguồn dữ liệu sẽ dùng
    if target_model and surrogate_model:
        paths = build_paths_from_models(target_model, surrogate_model)
        dataset = surrogate_model.rsplit("_", 2)[0] if surrogate_model else "?"
        st.sidebar.caption(f"📁 Dữ liệu: `{dataset}_raw_att.csv`")

st.sidebar.markdown("---")
st.sidebar.subheader("Thuật toán tấn công")

algorithm = st.sidebar.selectbox(
    "Thuật toán",
    options=["MIFGSM", "SIM", "VMIFGSM", "DGM"],
    index=0,
    help="MIFGSM: Momentum cơ bản | SIM: Scale-invariant | VMIFGSM: Variance-tuning | DGM: Diversity Gradient (mạnh nhất)",
)

iterations = st.sidebar.slider(
    "Iterations",
    min_value=1,
    max_value=20,
    value=7,
    help="Số bước lặp của thuật toán. Mặc định 7.",
)
step_size = st.sidebar.slider(
    "Step size",
    min_value=50,
    max_value=300,
    value=140,
    help="Kích thước bước nhiễu. Giá trị lớn hơn → tấn công mạnh hơn.",
)

copies = 5
dropout_rate = 0.2
if algorithm == "DGM":
    st.sidebar.subheader("Tham số DGM")
    copies = st.sidebar.slider(
        "Copies (N)",
        min_value=1,
        max_value=10,
        value=5,
        help="Số bản sao biến dạng để tính gradient trung bình.",
    )
    dropout_rate = st.sidebar.slider(
        "Dropout rate",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Xác suất dropout đặc trưng trong DGM.",
    )

# Main area
st.title("🛡️ SPTS NIDS Simulation Dashboard")
st.caption("Mô phỏng tấn công đối nghịch và đánh giá độ robust của NIDS dựa trên Deep Learning")

st.info(
    "💡 **Evasion Rate** = tỷ lệ % luồng tấn công bị NIDS phân loại nhầm là Benign. "
    "Giá trị cao nghĩa là tấn công hiệu quả, NIDS dễ bị bypass."
)

# Placeholders for results
st.subheader("📊 Kết quả")
evasion_placeholder = st.empty()
chart_placeholder = st.empty()

# Generate AAT button
if target_model and surrogate_model:
    if st.button("🚀 Generate AAT", type="primary", use_container_width=True):
        paths = build_paths_from_models(target_model, surrogate_model)
        if not os.path.exists(paths["fp_raw_att"]):
            st.error(f"Không tìm thấy file raw attack: `{paths['fp_raw_att']}`")
        elif not os.path.exists(paths["fp_model_s"]):
            st.error(f"Không tìm thấy surrogate model: `{paths['fp_model_s']}`")
        elif not os.path.exists(paths["fp_model_t"]):
            st.error(f"Không tìm thấy target model: `{paths['fp_model_t']}`")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress_cb(current: int, total: int, processed_rows: int, eta_sec):
                pct = current / total if total > 0 else 0
                progress_bar.progress(pct)
                if current < total:
                    phase = "Đang tạo AAT"
                    eta_str = f" | Còn ~{int(eta_sec // 60)} phút {int(eta_sec % 60)} giây" if eta_sec is not None and eta_sec > 0 else ""
                    status_text.caption(f"{phase} — batch {current}/{total} ({processed_rows} dòng){eta_str}")
                else:
                    status_text.caption("Đang đánh giá trên Target model...")

            try:
                raw_df, adv_df, metrics = generate_aat_from_data(
                    fp_raw_att=paths["fp_raw_att"],
                    fp_fea_s=paths["fp_fea_s"],
                    fp_minmax_s=paths["fp_minmax_s"],
                    fp_model_s=paths["fp_model_s"],
                    fp_model_t=paths["fp_model_t"],
                    fp_fea_t=paths["fp_fea_t"],
                    fp_minmax_t=paths["fp_minmax_t"],
                    algorithm=algorithm,
                    iterations=iterations,
                    step_size=step_size,
                    copies=copies,
                    dropout_rate=dropout_rate,
                    progress_callback=_progress_cb,
                )
                progress_bar.empty()
                status_text.empty()

                evasion_rate = metrics["evasion_rate"]
                level1_stats = metrics.get("level1_stats", {})
                attackable_features = metrics.get("attackable_features", [])

                st.session_state["last_run"] = {
                    "timestamp": datetime.now().isoformat(),
                    "target_model": target_model,
                    "surrogate_model": surrogate_model,
                    "algorithm": algorithm,
                    "iterations": iterations,
                    "step_size": float(step_size),
                    "copies": copies if algorithm == "DGM" else None,
                    "dropout_rate": dropout_rate if algorithm == "DGM" else None,
                    "raw_df": raw_df,
                    "adv_df": adv_df,
                    "metrics": metrics,
                    "attackable_features": attackable_features,
                }

                # Display evasion rate
                evasion_placeholder.metric(
                    "Evasion Rate (Bypass NIDS %)",
                    f"{evasion_rate:.1f}%",
                    delta=None,
                )
                evasion_placeholder.caption(
                    "Tỷ lệ luồng tấn công bị NIDS phân loại nhầm là Benign. Giá trị càng cao, tấn công càng hiệu quả."
                )

                # Level-1 feature distribution chart (matplotlib to avoid Arrow LargeUtf8)
                if level1_stats:
                    feats = [f for f in LEVEL1_FEATURES if f in level1_stats]
                    if feats:
                        before = [level1_stats[f]["before"]["mean"] for f in feats]
                        after = [level1_stats[f]["after"]["mean"] for f in feats]
                        x = range(len(feats))
                        fig, ax = plt.subplots(figsize=(8, 4))
                        w = 0.35
                        ax.bar([i - w/2 for i in x], before, w, label="Trước nhiễu")
                        ax.bar([i + w/2 for i in x], after, w, label="Sau nhiễu")
                        ax.set_xticks(x)
                        ax.set_xticklabels(feats, rotation=45, ha="right")
                        ax.legend()
                        ax.set_ylabel("Mean")
                        plt.tight_layout()
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", dpi=100)
                        buf.seek(0)
                        chart_placeholder.image(buf)
                        chart_placeholder.caption(
                            "Phân phối 4 đặc trưng Level-1 (SPTS): so sánh giá trị trung bình trước và sau khi thêm nhiễu."
                        )
                        plt.close()

                # Log to database
                log_simulation(
                    timestamp=datetime.now().isoformat(),
                    algorithm=algorithm,
                    iterations=iterations,
                    step_size=float(step_size),
                    target_model=target_model,
                    surrogate_model=surrogate_model,
                    evasion_rate=evasion_rate,
                    copies=copies if algorithm == "DGM" else None,
                    dropout_rate=dropout_rate if algorithm == "DGM" else None,
                )
                st.success("Đã tạo AAT và lưu log thành công!")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.exception(e)

# Compare + ExplainAI + History tabs
st.markdown("---")
tabs = st.tabs(["🧾 Compare AAT", "🧠 ExplainAI", "📋 Lịch sử"])

with tabs[0]:
    last_run = st.session_state.get("last_run")
    if not last_run:
        st.info("Chưa có kết quả để so sánh. Hãy bấm **Generate AAT** trước.")
    else:
        raw_df = last_run["raw_df"]
        adv_df = last_run["adv_df"]
        attackable = last_run.get("attackable_features", [])
        shared_cols = [c for c in attackable if c in raw_df.columns and c in adv_df.columns]
        if not shared_cols:
            shared_cols = [c for c in raw_df.columns if c in adv_df.columns]

        st.caption("So sánh AAT gốc (raw attack) và AAT được generate (adversarial) theo cùng thứ tự dòng.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("AAT gốc (raw)")
            st.dataframe(raw_df.head(50), use_container_width=True)
        with c2:
            st.subheader("AAT generate (adv)")
            st.dataframe(adv_df.head(50), use_container_width=True)

        raw_bytes = raw_df.to_csv(index=False).encode("utf-8")
        adv_bytes = adv_df.to_csv(index=False).encode("utf-8")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Tải raw CSV",
                data=raw_bytes,
                file_name="aat_raw.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Tải adv CSV",
                data=adv_bytes,
                file_name="aat_adv.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Thống kê thay đổi (Δ = adv - raw)")
        num_cols = [
            c
            for c in shared_cols
            if pd.api.types.is_numeric_dtype(raw_df[c]) and pd.api.types.is_numeric_dtype(adv_df[c])
        ]
        if not num_cols:
            st.warning("Không tìm thấy cột số để tính Δ.")
        else:
            delta = (adv_df[num_cols] - raw_df[num_cols]).astype("float64")
            summary = (
                pd.DataFrame(
                    {
                        "feature": num_cols,
                        "mean(|Δ|)": delta.abs().mean().values,
                        "mean(Δ)": delta.mean().values,
                        "std(Δ)": delta.std(ddof=0).values,
                        "changed_%": (delta.ne(0).mean() * 100).values,
                    }
                )
                .sort_values("mean(|Δ|)", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(summary.head(30), use_container_width=True)

with tabs[1]:
    last_run = st.session_state.get("last_run")
    if not last_run:
        st.info("Chưa có kết quả để giải thích. Hãy bấm **Generate AAT** trước.")
    else:
        raw_df = last_run["raw_df"]
        adv_df = last_run["adv_df"]
        attackable = last_run.get("attackable_features", [])
        shared_cols = [c for c in attackable if c in raw_df.columns and c in adv_df.columns]
        if not shared_cols:
            shared_cols = [c for c in raw_df.columns if c in adv_df.columns]

        num_cols = [
            c
            for c in shared_cols
            if pd.api.types.is_numeric_dtype(raw_df[c]) and pd.api.types.is_numeric_dtype(adv_df[c])
        ]
        if not num_cols:
            st.warning("Không tìm thấy cột số để explain.")
        else:
            delta = (adv_df[num_cols] - raw_df[num_cols]).astype("float64")
            importance = delta.abs().mean().sort_values(ascending=False)

            st.caption("ExplainAI (delta-based): feature nào bị điều chỉnh mạnh nhất được xếp theo mean(|Δ|).")
            top_k = st.slider(
                "Top-K features",
                min_value=5,
                max_value=min(50, len(importance)),
                value=min(15, len(importance)),
            )
            top = importance.head(top_k)

            fig, ax = plt.subplots(figsize=(9, max(3, int(top_k * 0.28))))
            ax.barh(list(reversed(top.index.tolist())), list(reversed(top.values.tolist())))
            ax.set_xlabel("mean(|Δ|)")
            ax.set_ylabel("feature")
            ax.set_title("Top features bị điều chỉnh")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120)
            buf.seek(0)
            st.image(buf)
            plt.close()

            st.subheader("Drill-down theo 1 flow")
            row_idx = st.number_input(
                "Row index",
                min_value=0,
                max_value=max(0, len(raw_df) - 1),
                value=0,
                step=1,
            )
            row_idx = int(row_idx)
            row_delta = (adv_df.loc[row_idx, num_cols] - raw_df.loc[row_idx, num_cols]).astype("float64")
            row_imp = row_delta.abs().sort_values(ascending=False).head(top_k)
            drill = pd.DataFrame(
                {
                    "feature": row_imp.index,
                    "raw": raw_df.loc[row_idx, row_imp.index].values,
                    "adv": adv_df.loc[row_idx, row_imp.index].values,
                    "Δ": row_delta.loc[row_imp.index].values,
                    "|Δ|": row_delta.loc[row_imp.index].abs().values,
                }
            ).reset_index(drop=True)
            st.dataframe(drill, use_container_width=True)

with tabs[2]:
    st.subheader("📋 Lịch sử mô phỏng")
    st.caption("Các lần chạy Generate AAT được lưu tự động. Bảng hiển thị mới nhất ở trên.")

    history = get_history()
    if history:
        df_history = pd.DataFrame(history)
        cols = ["timestamp", "algorithm", "iterations", "step_size", "target_model", "surrogate_model", "evasion_rate"]
        cols = [c for c in cols if c in df_history.columns]
        md = "| " + " | ".join(cols) + " |\n"
        md += "|" + "|".join([" --- " for _ in cols]) + "|\n"
        for _, row in df_history[cols].iterrows():
            md += "| " + " | ".join(str(v) for v in row) + " |\n"
        st.markdown(md)
    else:
        st.info("Chưa có bản ghi. Chạy Generate AAT để bắt đầu.")
