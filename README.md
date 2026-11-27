# AscendC 基础知识

## 部署网站

在线阅读 [算子研习册](https://kangdl.github.io/AscendC_Basic_Knowledge/)。

站点由 GitHub Pages 托管。推送到 `main` 或 `pages-deploy` 后，GitHub Actions 自动构建、校验并发布 `dist/`。

## Markdown 与网站架构

- `ops_learn_index.md` 的「第二卷 算子基础知识」提供正文。小节编号用于站内链接。
- `sites/operator-foundations/guides.md` 提供章节导读和路由，通过原稿章节标题与正文关联。
- 构建时将正文公式转为 MathML、Mermaid 转为静态图，并复制正文引用的本地图片和 HTML 附件。单独一行的 HTML 附件链接会嵌入正文。

## 代码架构

站点代码位于 `sites/operator-foundations/`。

| 文件 | 职责 |
| --- | --- |
| `build.mjs` | 解析 Markdown，生成 `dist/content.json` 和发布清单 `dist/manifest.json` |
| `render-diagrams.mjs` | 构建 Mermaid 图，普通图使用 SVG，含公式的图使用高清 PNG |
| `resource-path.mjs`、`verify-build.mjs` | 定义发布路径规则，校验产物与清单 |
| `src/` | HTML、CSS 和 JavaScript，负责页面路由、正文渲染与搜索 |
