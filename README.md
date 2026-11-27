# AscendC 基础知识

根目录的 `ops_learn_index.md` 是学习资料原稿。`sites/operator-foundations` 将原稿第二卷构建为可浏览、可搜索的「算子研习册」。

## 技术栈与本地构建

| 路径 | 用途 |
| --- | --- |
| `ops_learn_index.md` | 构建时读取的 Markdown 原稿 |
| `sites/operator-foundations/guides.md` | 章节摘要、导读、图解、速查表、练习和延伸阅读 |
| `sites/operator-foundations/src/` | 原生 HTML、CSS 和 JavaScript |
| `sites/operator-foundations/build.mjs` | 读取 Markdown、生成内容 JSON 并复制静态资源 |
| `sites/operator-foundations/dist/` | 构建产物，包含 `index.html`、`styles.css`、`app.js` 和 `content.json` |
| [`sites/operator-foundations/examples/content.json`](sites/operator-foundations/examples/content.json) | 一次本地构建的完整 JSON 示例，供查看章节和小节的字段 |
| `.github/workflows/deploy.yml` | GitHub Actions 构建、校验和发布配置 |

使用与 CI 一致的 Node.js 24，在仓库根目录运行以下命令。项目没有 npm 依赖或安装步骤。

```sh
node sites/operator-foundations/build.mjs
node sites/operator-foundations/preview.mjs
```

浏览器访问 [本地预览](http://127.0.0.1:8765/AscendC_Basic_Knowledge/)。前端通过 HTTP 读取 `content.json`，本地预览需要启动上述服务。修改后重新构建并刷新页面。产物目录已加入 Git 忽略规则。

## 编辑站点内容

1. 正文写在 `ops_learn_index.md` 的「第二卷 算子基础知识」下。保留小节标题开头的编号，已分享的小节链接依赖这些编号。
2. 章节导读写在 `sites/operator-foundations/guides.md`。沿用现有章节模板，让「原稿章节」对应原稿的二级标题文字。「路由」决定章节链接，「小节导读」按原稿中的小节编号关联。图解和速查表用两列 Markdown 表格，练习的题目和答案用空行分开，延伸阅读用 HTTP 或 HTTPS Markdown 链接。
3. 运行构建。脚本将正文、导读和搜索数据写入 `dist/content.json`，前端按当前路由渲染。内容修改提交 Markdown 源文件即可，JSON 由本地构建或 GitHub Actions 生成。

示例 JSON 保留生成时的内容快照，便于查看 `guides`（章节导读）和 `book`（正文与小节）的字段。GitHub Actions 每次从 Markdown 重新构建，并只上传 `dist/`，示例可以随文档需要单独更新。

正文段落、表格和代码可以在现有结构内编辑。构建脚本会校验第二卷标题、章节数量和小节编号，也会校验导读文件的必填栏目、表格列数及引用的小节编号。重命名原稿章节标题时，同步更新导读里的「原稿章节」。如果要调整章节层级或新增导读栏目，同步更新构建脚本。正文渲染器支持常用 Markdown 写法，新增语法时请在本地预览确认渲染效果。

## 首次部署

1. 用有相应仓库权限的账号打开 GitHub 仓库的 **Settings → Pages**，将 **Build and deployment → Source** 设为 **GitHub Actions**。仓库中的工作流会完成发布。
2. 如果仓库禁用了 Actions，在 **Settings → Actions → General** 启用 GitHub Actions，并允许工作流使用 GitHub 官方 `actions/*`。工作流已声明所需权限。
3. 将工作流、站点源码和 README 提交并推送或合并到 `main`。如果工作流已经在 `main` 上，启用 Pages 后可按下文手动运行。
4. 在 **Actions → Deploy to GitHub Pages** 确认构建和部署均成功，再用运行详情中的 `github-pages` 环境链接打开网站。当前仓库的默认项目站点地址为 [算子研习册](https://kangdl.github.io/AscendC_Basic_Knowledge/)。

[GitHub Pages 发布来源设置](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

## 自动更新与手动部署

- 推送或合并到 `main` 会触发构建和发布。`ops_learn_index.md` 的更新也会触发。
- 推送到 `pages-deploy` 也会发布到同一个站点，适用于独立部署。若 `github-pages` 环境限制了部署分支，仓库所有者需将 `pages-deploy` 加入允许列表。
- 工作流进入默认分支后，可在 **Actions → Deploy to GitHub Pages → Run workflow** 选择 `main` 或 `pages-deploy` 并运行。工作流仅接受这两个分支。
- 构建作业仅有仓库读取权限，发布作业仅有 Pages 写入和 OIDC 权限。工作流只上传经过固定文件清单校验的 `dist/`，并且无需配置 Secrets。第二卷内容、搜索索引和前端源码都会公开，提交前请确认其中没有密钥、敏感资料或本地配置。

## 故障排查

- **找不到工作流或无法手动运行**：确认工作流已进入默认分支 `main`，Actions 已启用，当前账号有运行权限。
- **构建失败**：在 **Actions** 展开 **Build site**。检查原稿是否仍有「第二卷 算子基础知识」标题，以及脚本预期的 7 章和数学附录。若 **Verify public files** 失败，检查生成文件是否符合公开文件清单。
- **发布失败**：确认 **Settings → Pages → Source** 为 **GitHub Actions**。如果 `github-pages` 环境已有分支限制或审批规则，确认它允许本次部署的分支并完成所需审批。
- **首页 404 或内容加载失败**：先打开部署详情给出的 URL，确认仓库子路径和末尾斜杠，例如 `/AscendC_Basic_Knowledge/`。在浏览器开发者工具的 Network 中检查 `app.js` 和 `content.json` 是否返回 200，同时检查 Console 中的错误。
- **章节链接 404**：使用带 `#` 的章节链接，例如 [Transformer 章节](https://kangdl.github.io/AscendC_Basic_Knowledge/#transformer/guide)。章节和小节由浏览器读取 URL 片段路由，GitHub Pages 始终只需提供同一个 `index.html`。

## 版本回滚

在 **Actions** 找到上一次成功的部署并记录提交 SHA。在用于部署的分支上通过新的提交撤销导致问题的改动，或恢复该版本的站点源码及 Markdown 原稿，必要时一并恢复构建脚本。推送或合并后会自动重新发布。部署成功后用首页和一个章节直达链接验证恢复结果。
