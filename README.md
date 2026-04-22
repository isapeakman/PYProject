入门学习
ecommerce-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── agent.py             # RAG 核心逻辑
│   ├── vector_store.py      # 向量数据库操作
│   ├── models.py            # 数据模型（Pydantic）
│   └── prompts.py           # 提示词模板
├── data/
│   ├── products.json        # 电商数据集
│   └── init_db.py           # 初始化向量库脚本
├── frontend/
│   └── streamlit_app.py     # Streamlit 界面
├── requirements.txt
└── .env                     # API Key 配置

![957d5410-b5f4-4aaf-a35a-7bab7516589c](picture/mermaid_20260422_da4346.png)
