"""``app.ingestion.loaders`` —— 七种格式的 loader。

每个 loader 都是**惰性**导入的（见 :mod:`app.ingestion.loaders.base` 的注册表），
所以只处理 TXT 的部署不会在 import 期拉起 PyMuPDF / python-docx / openpyxl。
"""
