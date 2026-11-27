# AscendC 基础知识

根目录的 `ops_learn_index.md` 是学习资料原稿。`sites/operator-foundations` 将原稿第二卷构建为可浏览、可搜索的「算子研习册」。

## 技术栈与本地构建

| 路径 | 用途 |
| --- | --- |
| `ops_learn_index.md` | 构建时读取的 Markdown 原稿 |
| `sites/operator-foundations/src/` | 原生 HTML、CSS 和 JavaScript |
| `sites/operator-foundations/build.mjs` | 通过 Node.js 内置模块解析第二卷、生成正文和搜索数据，并复制静态资源 |
| `sites/operator-foundations/dist/` | 构建产物，包含 `index.html`、`styles.css`、`app.js`、`guides.js` 和 `content.js` |
| `.github/workflows/deploy.yml` | GitHub Actions 构建、校验和发布配置 |

使用与 CI 一致的 Node.js 24，在仓库根目录运行以下命令。项目没有 npm 依赖或安装步骤。

```sh
node sites/operator-foundations/build.mjs
```

构建成功后可打开 `sites/operator-foundations/dist/index.html`。修改原稿或站点源码后重新构建即可预览。产物目录已加入 Git 忽略规则。

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
- **首页 404 或白屏**：先打开部署详情给出的 URL，确认仓库子路径和末尾斜杠，例如 `/AscendC_Basic_Knowledge/`。在浏览器开发者工具的 Network 中检查相对路径资源是否返回 200，尤其是 `content.js`。同时检查 Console 中的 JavaScript 错误。
- **章节链接 404**：使用带 `#` 的章节链接，例如 [Transformer 章节](https://kangdl.github.io/AscendC_Basic_Knowledge/#transformer/guide)。章节和小节由浏览器读取 URL 片段路由，GitHub Pages 始终只需提供同一个 `index.html`。

## 版本回滚

在 **Actions** 找到上一次成功的部署并记录提交 SHA。在用于部署的分支上通过新的提交撤销导致问题的改动，或恢复该版本的站点源码及 Markdown 原稿，必要时一并恢复构建脚本。推送或合并后会自动重新发布。部署成功后用首页和一个章节直达链接验证恢复结果。
