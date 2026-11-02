"""摄入层：文件 → 块与图片 → 分块 → 向量与索引。

模块边界
--------

::

    loaders/*       一种格式一个 loader，只负责「摊平」
        │           输出 RawBlock[] + RawImage[]，order 严格递增
        ▼
    parser          所有格式共用的一套归一化
        │           标题栈 → heading_path；图片锚定到上方最近的文本块
        ▼
    table_serializer  表格 → Markdown（独立原子块，不进正文切分流）
        ▼
    chunker         按标题边界切 300–800 token，code/table 整块保留
        ▼
    image_processor phash 去重 / 尺寸与信息量过滤 / VLM caption / 双向量化
        ▼
    pipeline        串起来，负责状态机、幂等、失败回滚

**刻意不做的事**：

* loader 不判断标题层级 —— 那会让同一套逻辑在七个文件里各写一遍；
* parser 不切分 —— 切分要看 token 预算，而 token 预算属于 ``chunking`` 配置；
* chunker 不碰数据库 —— 它返回纯 ``Chunk`` 列表，落库由 pipeline 负责；
* ``table_serializer`` 不依赖 tokenizer —— 需要 token 数时由调用方传函数进来。

这样每个模块都能在没有任何 API Key、没有任何数据库的情况下单独测试。
"""
