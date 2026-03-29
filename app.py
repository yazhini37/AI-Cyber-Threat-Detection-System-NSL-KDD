import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Cyber Threat Detection", layout="wide")

st.title("🚀 AI Cyber Threat Detection System")
st.markdown("### 🔐 Smart AI-based Intrusion Detection System")
st.write("Upload NSL-KDD dataset file and detect cyber attacks using AI")

# Load model
model = joblib.load("nsl_kdd_rf_pipeline.joblib")

# Upload file
uploaded_file = st.file_uploader("📂 Upload KDDTest+.txt file", type=["txt"])

if uploaded_file is not None:
    try:
        # Column names
        columns = [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
            "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
            "num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
            "dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate",
            "label","difficulty"
        ]

        # Read data
        data = pd.read_csv(uploaded_file, names=columns)

        st.subheader("📂 Uploaded Data")
        st.dataframe(data.head())

        # Run button
        if st.button("🚀 Run Detection"):

            # Prepare input
            X = data.drop(columns=["label", "difficulty"], errors="ignore")

            # Prediction
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

            # Add results
            data["Prediction"] = ["Attack 🚨" if p == 1 else "Normal ✅" for p in preds]
            data["Attack_Probability"] = probs

            st.subheader("🔍 Prediction Results")
            st.dataframe(data.head())

            # Summary
            attack_count = int(sum(preds))
            normal_count = int(len(preds) - attack_count)

            st.subheader("📊 Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"✅ Normal Traffic: {normal_count}")

            with col2:
                st.error(f"🚨 Attacks Detected: {attack_count}")

            # Accuracy (if label present)
            if "label" in data.columns:
                from sklearn.metrics import accuracy_score
                true_labels = data["label"].apply(lambda x: 0 if x == "normal" else 1)
                acc = accuracy_score(true_labels, preds)
                st.info(f"🎯 Model Accuracy: {acc:.2f}")

            # Chart
            st.subheader("📈 Traffic Distribution")

            fig, ax = plt.subplots()
            ax.bar(["Normal", "Attack"], [normal_count, attack_count])
            ax.set_ylabel("Count")
            ax.set_title("Network Traffic Analysis")

            st.pyplot(fig)

            # Download button
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results",
                csv,
                "cyber_threat_results.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")