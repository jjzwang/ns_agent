from typing import List, Dict, Any
class ChromaStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
    def connect(self):
        from chromadb import PersistentClient
        self.client = PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection("netsuite_kb")
    def add_documents(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
    def query(self, text: str, k: int = 40) -> List[Dict[str, Any]]:
        res = self.collection.query(query_texts=[text], n_results=k)
        results = []
        for i, doc in enumerate(res.get("documents", [[]])[0]):
            meta = res.get("metadatas", [[]])[0][i]
            results.append({"id": res.get("ids", [[]])[0][i], "text": doc, "metadata": meta, "score": float(res.get("distances", [[]])[0][i]) if res.get("distances") else 0.0})
        return results
