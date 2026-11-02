"""存储层。

四类存储，各有明确的所有权：

======================  ==========================  ==============================
模块                     存什么                       删知识库时谁负责
======================  ==========================  ==============================
``docstore``            知识库/文档/分块/图片记录/      :meth:`DocStore.delete_kb`
                        任务/会话/消息（SQLite）
``vectorstore``         文本向量 + 图像向量            :meth:`VectorStore.delete_collection`
                        （Chroma 或 pgvector）         ×2（``__text`` 与 ``__image``）
``bm25store``           分词后的稀疏索引               :meth:`BM25Store.drop_kb`
``blobstore``           图片原图（文件系统）            :meth:`BlobStore.delete_kb`
======================  ==========================  ==============================

**四者之间没有事务。** SQLite 的回滚不会撤销已经写进 Chroma 的向量，
也不会删掉已经落盘的图片。所以删除与重建都走「先标记 → 逐个清 → 最后删记录」
的顺序，中途失败留下的是一个**可被下次启动识别并续做**的中间态
（``KnowledgeBase.status == "deleting"``），而不是一个半死不活的知识库。

这个模块的 ``__init__`` **刻意不做任何 import**。原因：
``vectorstore_pg`` 顶层依赖 psycopg（在 ``[pg]`` extra 里），
``bm25store`` 顶层依赖 jieba，``vectorstore_chroma`` 顶层依赖 chromadb ——
在 ``__init__`` 里全量 import 会让「只想用其中一部分」的脚本
（比如只想读一下配置的 ``show_config.py``）被迫装齐所有依赖。
需要什么就 import 什么，装配统一走 :mod:`app.store.factory`。
"""
