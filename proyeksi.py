import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def show():
  st.title("📈 Proyeksi & Profiling Finansial")
  st.caption(
      "Proyeksi tren pengeluaran dan profil finansial berbasis model Machine"
      " Learning."
  )

  # Ambil artefak model dari session_state
  models = st.session_state.get("ml_models", {})
  lr_model = models.get("linear_reg")
  kmeans_model = models.get("kmeans")
  scaler = models.get("scaler")

  st.divider()

  # --- MODEL 1: REGRESI LINEAR (PROYEKSI PENGELUARAN) ---
  st.subheader("🔮 Estimasi Proyeksi Pengeluaran Bulan Depan")

  df = st.session_state.get("data_transaksi", pd.DataFrame()).copy()

  if not df.empty and "Date" in df.columns:
    df["Month_Str"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
    df_expense = df[df["Type"] == "EXPENSE"].copy()
  else:
    df_expense = pd.DataFrame()

  if not df_expense.empty and lr_model is not None:
    monthly_agg = (
        df_expense.groupby("Month_Str")["Amount"].sum().reset_index()
    )
    monthly_agg["Month_Index"] = np.arange(1, len(monthly_agg) + 1)

    # Prediksi bulan berikutnya menggunakan Regresi Linear
    next_month_index = len(monthly_agg) + 1
    raw_pred = lr_model.predict(np.array([[next_month_index]]))[0]

    # Penanganan safety agar nilai prediksi valid
    next_month_pred = float(max(0.0, float(raw_pred)))

    st.metric(
        "📊 Prediksi Model Regresi Linear (Bulan Depan)",
        f"Rp {next_month_pred:,.0f}",
    )

    fig_proj = go.Figure()

    # Line chart data historis
    fig_proj.add_trace(
        go.Scatter(
            x=monthly_agg["Month_Str"],
            y=monthly_agg["Amount"],
            mode="lines+markers",
            name="Pengeluaran Historis",
            line=dict(color="#1f77b4", width=3),
        )
    )

    # Line chart proyeksi masa depan
    months_labels = list(monthly_agg["Month_Str"]) + ["Proyeksi Bulan Depan"]
    proj_values = list(monthly_agg["Amount"]) + [next_month_pred]

    fig_proj.add_trace(
        go.Scatter(
            x=months_labels[-2:],
            y=proj_values[-2:],
            mode="lines+markers",
            name="Proyeksi Regresi Linear",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
        )
    )

    fig_proj.update_layout(
        title="Tren Pengeluaran Bulanan & Proyeksi Masa Depan",
        xaxis_title="Periode Bulan",
        yaxis_title="Nominal (Rupiah)",
    )
    st.plotly_chart(fig_proj, use_container_width=True)
  else:
    st.info(
        "Diperlukan data transaksi pengeluaran dan model Regresi Linear yang"
        " terkonfigurasi."
    )

  st.divider()

  # --- MODEL 2: K-MEANS CLUSTERING (PROFILING GAYA KEUANGAN) ---
  st.subheader("🎯 Profiling Gaya Keuangan (K-Means Clustering)")

  if not df.empty:
    total_inc = (
        float(df[df["Type"] == "INCOME"]["Amount"].sum())
        if "Type" in df.columns
        else 0.0
    )
    total_exp = (
        float(df[df["Type"] == "EXPENSE"]["Amount"].sum())
        if "Type" in df.columns
        else 0.0
    )
  else:
    total_inc, total_exp = 0.0, 0.0

  net_inc = total_inc - total_exp
  exp_ratio = (total_exp / total_inc) if total_inc > 0 else 1.0

  if kmeans_model is not None and scaler is not None:
    user_features = np.array([[net_inc, exp_ratio]])
    scaled_features = scaler.transform(user_features)
    cluster_id = int(kmeans_model.predict(scaled_features)[0])

    # Mapping nama cluster berbasis murni hasil prediksi K-Means (Cluster ID)
    cluster_mapping = {
        0: {
            "label": "Normal",
            "desc": (
                "Rasio pengeluaran Anda seimbang (70% - 90%). Arus kas cukup"
                " teratur."
            ),
            "style": "info",
            "icon": "🟡",
        },
        1: {
            "label": "Hemat",
            "desc": (
                "Rasio pengeluaran Anda di bawah 70%. Pengelolaan keuangan"
                " sangat baik dan hemat!"
            ),
            "style": "success",
            "icon": "🟢",
        },
        2: {
            "label": "Boros",
            "desc": (
                "Rasio pengeluaran melebihi 90% dari pemasukan. Tingkat"
                " pengeluaran tinggi/boros!"
            ),
            "style": "warning",
            "icon": "🔴",
        },
    }

    # Ambil detail metadata cluster berdasarkan ID hasil pelatihan K-Means
    info_cluster = cluster_mapping.get(
        cluster_id,
        {
            "label": "Unknown",
            "desc": "Profil keuangan tidak terdefinisi.",
            "style": "info",
            "icon": "⚪",
        },
    )

    cluster_label = info_cluster["label"]
    cluster_desc = info_cluster["desc"]
    cluster_style = info_cluster["style"]
    badge_icon = info_cluster["icon"]

    # Tampilan Hasil Profiling ML
    st.markdown(f"### {badge_icon} Profil: **{cluster_label}**")

    if cluster_style == "success":
      st.success(cluster_desc)
    elif cluster_style == "warning":
      st.warning(cluster_desc)
    else:
      st.info(cluster_desc)

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Net Income Bulanan", f"Rp {net_inc:,.0f}")
    mcol2.metric("Rasio Pengeluaran", f"{exp_ratio * 100:.1f}%")
    mcol3.metric(
        "Hasil Klaster ML", f"{cluster_label} (Cluster #{cluster_id})"
    )
  else:
    st.info(
        "Model K-Means Clustering atau Scaler belum dimuat dari folder"
        " artifacts."
    )