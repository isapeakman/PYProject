import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="ecommerce-agent", page_icon="🛍️")
st.title("🛍️ Ecommerce Agent")
st.caption("基于 FastAPI + ChromaDB 的电商问答示例")

query = st.text_input("请输入你的问题", placeholder="例如：有什么降噪耳机？")
top_k = st.slider("召回条数", min_value=1, max_value=5, value=3)

if st.button("提问"):
    if not query.strip():
        st.warning("请输入问题后再提交。")
    else:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/ask",
                json={"query": query, "top_k": top_k},
                timeout=90,
            )
            resp.raise_for_status()
            st.success(resp.json().get("answer", "无返回内容"))
        except requests.RequestException as exc:
            st.error(f"请求失败：{exc}")
