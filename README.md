入门学习 <br>
app/ <br>
├── __init__.py          # app package <br>   
├── main.py              # FastAPI 入口 <br>
├── agent.py             # RAG 核心逻辑 <br>
├── vector_store.py      # 向量数据库操作 <br>
├── models.py            # 数据模型（Pydantic） <br>
    └── prompts.py           # 提示词模板 <br>
data/ <br>
├── products.json        # 电商数据集 <br>
    └── init_db.py           # 初始化向量库脚本 <br>
frontend/<br>
└── streamlit_app.py     # Streamlit 界面<br>
requirements.txt <br>
.env                     # API Key 配置 <br>

![957d5410-b5f4-4aaf-a35a-7bab7516589c](picture/mermaid_20260422_da4346.png)
