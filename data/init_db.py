import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # 允许通过 `python data/init_db.py` 直接运行脚本
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vector_store import get_vector_store

DATA_FILE = Path(__file__).parent / "products.json"


def build_text(item: dict) -> str:
    """把结构化商品数据拼成便于向量化的文本。"""
    feature_text = "，".join(item.get("features", []))
    return f"{item['name']}，价格{item['price']}元，{feature_text}"


def main() -> None:
    """初始化向量库，并跳过已经写入过的商品。"""
    products = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    vector_store = get_vector_store()

    existing = vector_store.get(include=[])
    existing_ids = set(existing.get("ids", []))

    docs = []
    ids = []
    metadatas = []
    for item in products:
        if item["id"] in existing_ids:
            continue
        ids.append(item["id"])
        docs.append(build_text(item))
        metadatas.append({"name": item["name"], "price": item["price"]})

    if not ids:
        print("向量库已是最新，无需初始化。")
        return

    vector_store.add_texts(texts=docs, ids=ids, metadatas=metadatas)
    print(f"已写入 {len(ids)} 条商品数据。")


if __name__ == "__main__":
    main()
