import streamlit as st
from utils.api import upload_pdfs_api


def render_uploader():

    st.markdown("""
    <p style="color:#7eb3d8; font-size:0.88rem; font-weight:600; margin-bottom:10px;">
        📄 Upload Medical Documents
    </p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more medical PDF reports to query"
    )

    # Show selected files info
    if uploaded_files:
        st.markdown(f"""
        <div style="
            background-color: #0d2137;
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 10px 14px;
            margin: 10px 0;
        ">
            <p style="color:#58b0e0; font-size:0.82rem; margin:0 0 6px 0;">
                📎 {len(uploaded_files)} file(s) selected:
            </p>
        """, unsafe_allow_html=True)

        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.markdown(f"""
            <p style="color:#c9d1d9; font-size:0.8rem; margin: 2px 0;">
                • {f.name} <span style="color:#4a5568;">({size_kb} KB)</span>
            </p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Upload button
    if st.button("🚀 Upload & Process", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("⚠️ Please select at least one PDF file first.")
            return

        progress = st.progress(0, text="Preparing files...")
        status   = st.empty()

        try:
            progress.progress(30, text="Uploading to server...")
            status.info("⏳ Processing... This may take a moment.")

            # ✅ Bug fix: was "upload_pdfs_api" missing () and argument
            response = upload_pdfs_api(uploaded_files)

            progress.progress(90, text="Finalizing...")

            if response.status_code == 200:
                progress.progress(100, text="Done!")
                status.success(
                    f"✅ {len(uploaded_files)} file(s) uploaded successfully! "
                    "You can now ask questions."
                )
                st.session_state.docs_uploaded = True
            else:
                status.error(f"❌ Upload failed: {response.text}")
                progress.empty()

        except Exception as e:
            status.error(f"❌ Error: {str(e)}")
            progress.empty()

    # Status indicator
    if st.session_state.get("docs_uploaded"):
        st.markdown("""
        <div style="
            background-color: #0a2e1a;
            border: 1px solid #1a5c34;
            border-radius: 8px;
            padding: 8px 14px;
            margin-top: 10px;
        ">
            <p style="color:#3ddc84; font-size:0.82rem; margin:0;">
                ✅ Documents ready — ask your questions!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background-color: #1a1f2e;
            border: 1px solid #2a2f3e;
            border-radius: 8px;
            padding: 8px 14px;
            margin-top: 10px;
        ">
            <p style="color:#4a5568; font-size:0.82rem; margin:0;">
                👆 Upload a PDF to get started
            </p>
        </div>
        """, unsafe_allow_html=True)