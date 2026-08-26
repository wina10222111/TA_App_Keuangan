import plotly.express as px
import pandas as pd
import streamlit as st


def show():
    st.title("📊 Dashboard Utama Keuangan")
    st.caption("Ringkasan arus kas dan alokasi pengeluaran bulanan.")

    df = st.session_state["data_transaksi"].copy()

    if df.empty:
        st.info("Belum ada data transaksi. Silakan masukkan data di menu Pencatatan.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month_Str"] = df["Date"].dt.to_period("M").astype(str)

    available_months = sorted(df["Month_Str"].unique(), reverse=True)

    st.divider()

    col_title, col_filter = st.columns([2, 1])
    with col_title:
        st.subheader("📅 Filter Periode Transaksi")
    with col_filter:
        selected_month = st.selectbox(
            "Pilih Bulan:", options=["Semua Periode"] + available_months
        )

    if selected_month != "Semua Periode":
        df_filtered = df[df["Month_Str"] == selected_month].copy()
    else:
        df_filtered = df.copy()

    total_income = df_filtered[df_filtered["Type"] == "INCOME"]["Amount"].sum()
    total_expense = df_filtered[df_filtered["Type"] == "EXPENSE"]["Amount"].sum()
    net_income = total_income - total_expense

    expense_ratio = (
        (total_expense / total_income * 100) if total_income > 0 else 100.0
    )

    if expense_ratio < 70:
        status_label = "🟢 HEMAT / SEHAT"
        status_color = "#28a745"
        status_msg = f"Rasio pengeluaran **{expense_ratio:.1f}%** dari pemasukan. Arus kas dalam kondisi baik."
    elif 70 <= expense_ratio <= 90:
        status_label = "🟡 NORMAL / WASPADA"
        status_color = "#ffc107"
        status_msg = f"Rasio pengeluaran **{expense_ratio:.1f}%** dari pemasukan. Perhatikan pengeluaran opsional."
    else:
        status_label = "🔴 BOROS / KRITIS"
        status_color = "#dc3545"
        status_msg = f"Rasio pengeluaran mencapai **{expense_ratio:.1f}%** dari pemasukan."

    st.markdown(
        f"""
    <div style="background-color: {status_color}15; border-left: 6px solid {status_color}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: {status_color};">{status_label}</h3>
        <p style="margin: 5px 0 0 0; color: #333; font-size: 15px;">{status_msg}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Hanya menampilkan 3 Metrik Arus Kas Utama
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Pemasukan", f"Rp {total_income:,.0f}")
    col2.metric("💸 Total Pengeluaran", f"Rp {total_expense:,.0f}")
    col3.metric(
        "💵 Net Income (Sisa)",
        f"Rp {net_income:,.0f}",
        delta=f"{net_income:,.0f}",
        delta_color="normal" if net_income >= 0 else "inverse",
    )

    st.divider()

    st.subheader(f"📌 Distribusi Pengeluaran per Kategori ({selected_month})")

    df_expense = df_filtered[df_filtered["Type"] == "EXPENSE"]

    if not df_expense.empty:
        chart_col1, chart_col2 = st.columns(2)

        cat_summary = (
            df_expense.groupby("Category")["Amount"]
            .sum()
            .reset_index()
            .sort_values(by="Amount", ascending=False)
        )

        with chart_col1:
            fig_donut = px.pie(
                cat_summary,
                values="Amount",
                names="Category",
                title="Persentase Kategori Pengeluaran",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart_col2:
            fig_bar = px.bar(
                cat_summary,
                x="Category",
                y="Amount",
                title="Nominal Pengeluaran per Kategori",
                color="Category",
                text_auto=".2s",
            )
            fig_bar.update_layout(showlegend=False, yaxis_title="Rupiah")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(f"Belum ada data pengeluaran pada periode {selected_month}.")