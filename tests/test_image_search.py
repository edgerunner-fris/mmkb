"""以图搜图的离线测试。

覆盖三件事：

1. 图片必须走多模态空间、查 ``{kb}__image``，不能被文本 provider 顶替；
2. 知识库快照（维度/模型/空间）不一致时必须报错，而不是返回一批错误排序；
3. 空 collection 直接跳过，不花一次 embedding 调用。
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pydantic import SecretStr

from app.config import (
    AppSettings,
    BlobSettings,
    BM25Settings,
    DocStoreSettings,
    ImageEmbeddingSettings,
    LLMSettings,
    PathSettings,
    RerankSettings,
    Settings,
    TextEmbeddingSettings,
    VectorStoreSettings,
    VisionSettings,
)
from app.core.errors import ConfigurationError, ProviderResponseError, ValidationError
from app.core.schemas import KnowledgeBase
from app.main import create_app
from app.providers.base import EmbeddingProvider, EmbeddingResult, PreparedImage
from app.services.container import AppContainer
from app.services.search_service import SearchService
from app.store.vectorstore import VectorHit


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app=AppSettings(env="dev", log_level="WARNING"),
        llm=LLMSettings(api_key=SecretStr("")),
        vision=VisionSettings(enabled=False),
        text_embed=TextEmbeddingSettings(api_key=SecretStr("")),
        image_embed=ImageEmbeddingSettings(api_key=SecretStr("")),
        rerank=RerankSettings(api_key=SecretStr("")),
        paths=PathSettings(data_dir=tmp_path),
        docstore=DocStoreSettings(sqlite_path=tmp_path / "test.db"),
        blob=BlobSettings(images_dir=tmp_path / "images"),
        bm25=BM25Settings(index_dir=tmp_path / "bm25", persist=False),
        vectorstore=VectorStoreSettings(chroma_dir=tmp_path / "chroma"),
    )


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _kb(**overrides: Any) -> KnowledgeBase:
    payload: dict[str, Any] = {
        "kb_id": "kb_test",
        "name": "测试库",
        "text_collection": "kb_test__text",
        "image_collection": "kb_test__image",
        "text_dim": 1024,
        "image_dim": 8,
        "llm_model": "deepseek-flash",
        "text_embed_model": "text-embedding-v4",
        "image_embed_model": "qwen3-vl-embedding",
        "rerank_model": "gte-rerank-v2",
    }
    payload.update(overrides)
    return KnowledgeBase.model_validate(payload)


class _ImageProvider(EmbeddingProvider):
    name = "fake-multimodal"
    model = "qwen3-vl-embedding"
    space = "image"
    dim = 8

    def __init__(self) -> None:
        self.calls = 0
        self.received: list[Any] = []
        self.vectors: list[list[float]] = [[0.1] * 8]

    @property
    def supports_image(self) -> bool:
        return True

    async def embed_texts(self, texts: Any) -> EmbeddingResult:
        raise AssertionError("以图搜图不能调用 embed_texts")

    async def embed_images(self, images: Any) -> EmbeddingResult:
        self.calls += 1
        self.received = list(images)
        return EmbeddingResult(vectors=self.vectors, model=self.model, dim=self.dim)

    async def embed_fused(self, texts: Any) -> EmbeddingResult:
        raise AssertionError("以图搜图不能调用 embed_fused")


class _DocStore:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.requested_doc_ids: list[str] = []

    async def require_kb(self, kb_id: str) -> KnowledgeBase:
        assert kb_id == self.kb.kb_id
        return self.kb

    async def get_document(self, doc_id: str) -> Any:
        self.requested_doc_ids.append(doc_id)
        return SimpleNamespace(doc_id=doc_id, source_name="示例文档.pdf")

    async def get_images(self, image_ids: list[str]) -> list[Any]:
        return [
            SimpleNamespace(image_id=image_id, url=f"/static/images/kb_test/{image_id}.png")
            for image_id in image_ids
        ]


class _VectorStore:
    def __init__(self, count: int = 1) -> None:
        self._count = count
        self.queries: list[dict[str, Any]] = []

    async def count(self, name: str) -> int:
        assert name == "kb_test__image"
        return self._count

    async def query(
        self,
        name: str,
        vector: list[float],
        *,
        top_k: int = 10,
        space: str | None = None,
        **kwargs: Any,
    ) -> list[VectorHit]:
        self.queries.append(
            {"name": name, "vector": vector, "top_k": top_k, "space": space}
        )
        return [
            VectorHit(
                id="img_1",
                score=0.91,
                document="架构图",
                metadata={
                    "image_id": "img_1",
                    "kb_id": "kb_test",
                    "doc_id": "doc_1",
                    "url": "/static/images/kb_test/img_1.png",
                    "caption": "架构图：三路召回与融合",
                },
            )
        ]


def _container(
    kb: KnowledgeBase,
    provider: _ImageProvider | None = None,
    count: int = 1,
) -> tuple[SimpleNamespace, _ImageProvider, _VectorStore, _DocStore]:
    image_provider = provider or _ImageProvider()
    docstore = _DocStore(kb)
    vectorstore = _VectorStore(count=count)
    container = SimpleNamespace(
        image_embedding=image_provider,
        docstore=docstore,
        vectorstore=vectorstore,
        require_image_embedding=lambda: image_provider,
    )
    return container, image_provider, vectorstore, docstore


async def test_image_search_uses_image_space_and_hydrates() -> None:
    kb = _kb()
    container, provider, vectorstore, docstore = _container(kb)
    outcome = await SearchService(container).image_search(
        "kb_test", _png(), top_k=5
    )

    assert provider.calls == 1
    assert isinstance(provider.received[0], PreparedImage)
    assert vectorstore.queries == [
        {"name": "kb_test__image", "vector": [0.1] * 8, "top_k": 5, "space": "image"}
    ]
    assert outcome["count"] == 1
    assert outcome["reranked"] is False
    assert outcome["routes"][0]["route"] == "dense_image"
    item = outcome["items"][0]
    assert item["item_type"] == "image"
    assert item["item_id"] == "img_1"
    assert item["source_name"] == "示例文档.pdf"
    assert item["image_url"] == "/static/images/kb_test/img_1.png"
    assert docstore.requested_doc_ids == ["doc_1"]


async def test_empty_image_collection_skips_embedding_call() -> None:
    kb = _kb()
    container, provider, vectorstore, _ = _container(kb, count=0)
    outcome = await SearchService(container).image_search("kb_test", _png())

    assert provider.calls == 0
    assert vectorstore.queries == []
    assert outcome["count"] == 0
    assert outcome["routes"][0]["skipped"] is True
    assert "为空" in outcome["routes"][0]["note"]


async def test_snapshot_dimension_mismatch_is_fatal() -> None:
    kb = _kb(image_dim=2560)
    container, provider, _, _ = _container(kb)
    with pytest.raises(Exception) as excinfo:
        await SearchService(container).image_search("kb_test", _png())
    assert "维度" in str(excinfo.value)
    # 快照不一致时不能先花钱编码，再发现空间不对
    assert provider.calls == 0


async def test_snapshot_model_mismatch_is_fatal() -> None:
    kb = _kb(image_embed_model="another-model")
    container, provider, _, _ = _container(kb)
    with pytest.raises(Exception) as excinfo:
        await SearchService(container).image_search("kb_test", _png())
    assert "模型" in str(excinfo.value)
    assert provider.calls == 0


async def test_empty_bytes_rejected() -> None:
    kb = _kb()
    container, _, _, _ = _container(kb)
    with pytest.raises(ValidationError):
        await SearchService(container).image_search("kb_test", b"")


async def test_non_image_bytes_rejected() -> None:
    kb = _kb()
    container, provider, _, _ = _container(kb)
    with pytest.raises(ValidationError):
        await SearchService(container).image_search("kb_test", b"not an image")
    assert provider.calls == 0


async def test_empty_vector_is_provider_error() -> None:
    kb = _kb()
    provider = _ImageProvider()
    provider.vectors = []
    container, _, _, _ = _container(kb, provider=provider)
    with pytest.raises(ProviderResponseError):
        await SearchService(container).image_search("kb_test", _png())


def test_image_search_route_is_registered(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    paths = app.openapi()["paths"]
    assert "/api/search/image" in paths
    operation = paths["/api/search/image"]["post"]
    # 上传接口必须是 multipart，否则前端/CLI 传文件会 422
    assert "multipart/form-data" in operation["requestBody"]["content"]


def test_missing_image_provider_reports_its_own_cause() -> None:
    # 只构造方法所需的状态，避免真的去建 Chroma / SQLite
    container = object.__new__(AppContainer)
    container._image_embedding = None
    container._provider_errors = [  # type: ignore[attr-defined]
        "LLM_API_KEY 未配置，无法构造对话模型",
        "TEXT_EMBED_API_KEY 未配置，无法构造文本向量模型",
    ]
    with pytest.raises(ConfigurationError) as excinfo:
        container.require_image_embedding()
    message = str(excinfo.value)
    # 不能把「问答缺 Key」当成以图搜图失败的原因
    assert "IMAGE_EMBED_API_KEY" in message
    assert "LLM_API_KEY" not in message
