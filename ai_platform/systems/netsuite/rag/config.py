from dataclasses import dataclass
import yaml
@dataclass
class RAGConfig:
    pdf_directory: str
    persist_directory: str
    gemini_model: str = "models/gemini-1.5-pro"
    gemini_embedding_model: str = "models/text-embedding-004"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    enable_toc_extraction: bool = True
    retrieval_k: int = 8
    k_vec: int = 40
    k_bm: int = 200
    fusion_weight_vec: float = 0.6
    fusion_weight_bm25: float = 0.4
    max_requests_per_minute: int = 300
    max_requests_per_day: int = 10000
    embedding_batch_delay: float = 0.1
def load_config(path: str = "config.yaml") -> RAGConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RAGConfig(**data)
