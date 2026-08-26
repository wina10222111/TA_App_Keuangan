import pandas as pd
import streamlit as st


def show():
    st.title("📝 Pencatatan Transaksi & Auto-Categorization")
    st.caption("Input transaksi harian. Kategori otomatis ditentukan oleh model machine learning.")

    models = st.session_state.get("ml_models", {})
    clean_func = st.session_state.get("clean_text_func", lambda x: x)

    rf_model = models.get("rf_semantic")
    embedder = models.get("embedder")
    tfidf = models.get("tfidf")
    nlp_type = models.get("nlp_type", "semantic")

    st.divider()

    with st.form("form_transaksi", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            tgl = st.date_input("Tanggal Transaksi", value=pd.Timestamp.now())
            tipe = st.selectbox("Tipe Transaksi", ["EXPENSE", "INCOME"])
            nominal = st.number_input(
                "Nominal (Rp)", min_value=1000, value=50000, step=5000
            )

        with col2:
            judul = st.text_input(
                "Deskripsi / Judul Transaksi",
                placeholder="Misal: Beli token PLN / Makan bakso bersama teman",
            )

            predicted_category = "Lain-lain"
            if judul and tipe == "EXPENSE" and rf_model is not None:
                cleaned_title = clean_func(judul)
                if cleaned_title:
                    if nlp_type in ["semantic", "classifier"] and embedder is not None:
                        title_vector = embedder.encode([cleaned_title])
                        predicted_category = rf_model.predict(title_vector)[0]
                    elif nlp_type == "tfidf" and tfidf is not None:
                        title_vector = tfidf.transform([cleaned_title])
                        predicted_category = rf_model.predict(title_vector)[0]
                    else:
                        predicted_category = rf_model.predict([cleaned_title])[0]

            if tipe == "INCOME":
                predicted_category = "Pendapatan"

            st.write("🤖 **Hasil Prediksi Model Machine Learning:**")
            st.info(f"🏷️ Kategori Terdeteksi: **{predicted_category}**")

        submitted = st.form_submit_button(
            "💾 Simpan Transaksi", use_container_width=True
        )

        if submitted:
            if not judul.strip():
                st.warning("⚠️ Harap masukkan deskripsi transaksi terlebih dahulu.")
            else:
                kategori_final = predicted_category

                new_data = pd.DataFrame([{
                    "Date": pd.to_datetime(tgl),
                    "Title": judul,
                    "Type": tipe,
                    "Amount": nominal,
                    "Category": kategori_final,
                }])

                st.session_state["data_transaksi"] = pd.concat(
                    [st.session_state["data_transaksi"], new_data],
                    ignore_index=True,
                )

                st.success(f"✅ Transaksi '{judul}' berhasil disimpan ke kategori '{kategori_final}'!")

    st.divider()

    st.subheader("📋 Riwayat Transaksi Tersimpan")
    df_display = st.session_state["data_transaksi"].copy()

    if not df_display.empty:
        df_display = df_display.sort_index(ascending=False)
        df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%Y-%m-%d")
        df_display["Amount"] = df_display["Amount"].apply(lambda x: f"Rp {x:,.0f}")

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Belum ada riwayat transaksi.")
