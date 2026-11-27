const main = document.querySelector('main');
const { book, guides } = await loadContent();
const chapterNav = document.querySelector('#chapter-nav');
const sidebar = document.querySelector('#sidebar');
const navToggle = document.querySelector('.nav-toggle');
const scrim = document.querySelector('.nav-scrim');
const dialog = document.querySelector('#search-dialog');
const searchInput = dialog.querySelector('input');
const searchResults = dialog.querySelector('#search-results');
const mobileNav = matchMedia('(max-width: 850px)');
let route;
let searchOpener;
const expandedBranches = new Set();

async function loadContent() {
  try {
    const response = await fetch(new URL('./content.json', import.meta.url));
    if (!response.ok) throw new Error('content.json HTTP ' + response.status);
    return await response.json();
  } catch (error) {
    main.innerHTML = '<div class="home"><p role="alert">内容加载失败，请刷新页面重试。</p></div>';
    throw error;
  }
}

const contents = book.map((entry) => {
  const byId = new Map(entry.sections.map((section) => [section.id, section]));
  const children = new Map();
  for (const section of entry.sections) {
    if (!children.has(section.parentId)) children.set(section.parentId, []);
    children.get(section.parentId).push(section);
  }
  return { byId, children };
});

const escape = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const icon = (name) => '<span aria-hidden="true" class="glyph glyph-' + name + '"><i></i><i></i><i></i><i></i></span>';
const chapterHref = (id, section = '') => '#' + id + (section ? '/' + section : '');
const chapterNumber = (guide) => guide.number ? '第 ' + Number(guide.number) + ' 章' : '附录';

function sectionTrail(index, sectionId) {
  const trail = [];
  const byId = contents[index]?.byId;
  let section = byId?.get(sectionId);
  while (section) {
    trail.unshift(section);
    section = byId.get(section.parentId);
  }
  return trail;
}

function readRoute() {
  const [id, section] = location.hash.slice(1).split('/');
  if (id === 'curriculum') return { id, section: '', index: -1, trail: [] };
  const index = guides.findIndex((guide) => guide.id === id);
  const trail = sectionTrail(index, section);
  return { id: index === -1 ? 'overview' : id, section, index, trail, page: trail.findLast((item) => item.isPage) };
}

function setNav(open) {
  const visible = open && mobileNav.matches;
  document.body.classList.toggle('nav-open', visible);
  navToggle.setAttribute('aria-expanded', visible);
  navToggle.setAttribute('aria-label', visible ? '关闭章节目录' : '打开章节目录');
  scrim.hidden = !visible;
  sidebar.inert = mobileNav.matches && !visible;
}

function sidebarTree(index, parentId = null) {
  const sections = contents[index].children.get(parentId);
  if (!sections) return '';
  const guide = guides[index];
  return '<ul class="subnav">' + sections.map((section) => {
    const { id, title } = section;
    const branchId = 'branch-' + guide.id + '-' + id;
    const hasChildren = contents[index].children.has(id);
    const open = expandedBranches.has(branchId) || route.trail.includes(section);
    const current = route.section === id ? (section.isPage ? 'page' : 'location') : (route.page === section ? 'page' : '');
    return '<li><div class="subnav-row"><a href="' + chapterHref(guide.id, id) + '"' +
      (current ? ' aria-current="' + current + '"' : '') + '>' + escape(title) + '</a>' +
      (hasChildren ? '<button type="button" class="tree-toggle" aria-label="' + escape(title) + '的小节" aria-controls="' +
        branchId + '" aria-expanded="' + open + '"><i aria-hidden="true"></i></button>' : '') + '</div>' +
      (hasChildren ? '<div id="' + branchId + '"' + (open ? '' : ' hidden') + '>' + sidebarTree(index, id) + '</div>' : '') + '</li>';
  }).join('') + '</ul>';
}

function sidebarChapter(guide, index) {
  const active = route.index === index;
  return '<a class="side-link' + (active ? ' chapter-active' : '') + '" href="#' + guide.id + '"' +
    (active && !route.page ? ' aria-current="page"' : '') + '>' +
    (guide.number ? '<span class="side-number">' + guide.number + '</span>' : icon('math')) +
    '<span>' + escape(guide.title) + '</span></a>' + (active ? sidebarTree(index) : '');
}

function showSidebar() {
  const chapters = guides.slice(0, -1).map((guide, index) =>
    '<li>' + sidebarChapter(guide, index) + '</li>').join('');
  const appendixIndex = guides.length - 1;
  chapterNav.innerHTML = '<a class="side-link side-home" href="#overview"' +
    (route.index < 0 ? ' aria-current="page"' : '') + '>' + icon('book') + '首页</a>' +
    '<p class="nav-caption">章节目录</p><ul>' + chapters + '</ul>' +
    '<div class="nav-appendix">' + sidebarChapter(guides[appendixIndex], appendixIndex) + '</div>';
  const current = chapterNav.querySelector('.subnav [aria-current="location"]') ??
    chapterNav.querySelector('.subnav [aria-current="page"]') ?? chapterNav.querySelector('[aria-current="page"]');
  if (current) {
    const item = current.getBoundingClientRect();
    const rail = sidebar.getBoundingClientRect();
    if (item.bottom > rail.bottom) sidebar.scrollTop += item.bottom - rail.bottom + 16;
    if (item.top < rail.top) sidebar.scrollTop -= rail.top - item.top + 16;
  }
  document.querySelectorAll('[data-top]').forEach((link) => {
    const id = route.index >= 0 && route.id !== 'math' ? 'curriculum' : route.id;
    if (link.dataset.top === id) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}

function attentionDemo() {
  return '<figure class="attention">' +
    '<div class="figure-head"><div><div class="figure-kicker">交互图解</div><h2>因果注意力</h2></div><span class="shape">4 × 4</span></div>' +
    '<div class="figure-content"><div class="matrix-wrap"><span class="matrix-title">键位置 K</span><div class="matrix" role="group" aria-label="因果掩码矩阵"></div></div>' +
    '<div class="matrix-side"><span class="query-label">查询位置 Q</span><div class="query-buttons" role="group" aria-label="选择查询 token">' +
    [1, 2, 3, 4].map((n) => '<button type="button" data-token="' + n + '" aria-label="第 ' + n + ' 个 token">' + n + '</button>').join('') +
    '</div><p class="query-explanation" aria-live="polite"></p></div></div>' +
    '<figcaption class="legend"><span><i></i>可见位置</span><span><i></i>被掩码屏蔽</span></figcaption></figure>';
}

function updateAttention(figure, token) {
  const cells = ['<span></span>'];
  for (let column = 1; column <= 4; column++) cells.push('<span class="matrix-index">t' + column + '</span>');
  for (let row = 1; row <= 4; row++) {
    cells.push('<span class="matrix-index' + (row === token ? ' current' : '') + '">t' + row + '</span>');
    for (let column = 1; column <= 4; column++) {
      const visible = column <= row;
      cells.push('<span class="matrix-cell ' + (visible ? 'allowed' : 'masked') + (row === token ? ' current-row' : '') +
        '" aria-label="查询 t' + row + '、键 t' + column + '，' + (visible ? '可见' : '屏蔽') + '">' + (visible ? '●' : '–') + '</span>');
    }
  }
  figure.querySelector('.matrix').innerHTML = cells.join('');
  figure.querySelectorAll('[data-token]').forEach((button) =>
    button.setAttribute('aria-pressed', Number(button.dataset.token) === token));
  figure.querySelector('.query-explanation').textContent = token === 1 ? 't1 只能看到自己。' :
    't' + token + ' 可以看到 t1 到 t' + token + '。';
}

function showRoadmap() {
  const stages = [
    ['理解模型计算', guides.slice(0, 2)],
    ['学习算子实现', guides.slice(2, 4)],
    ['理解硬件与优化', guides.slice(4, -1)],
  ];
  const links = (chapters) => chapters.map((guide) => '<a href="#' + guide.id + '">' + escape(guide.title) + '</a>').join('');
  return '<section class="roadmap" aria-labelledby="roadmap-title"><h2 id="roadmap-title">学习路线</h2><ol>' +
    stages.map(([label, chapters]) => '<li><strong>' + label + '</strong><div>' + links(chapters) + '</div></li>').join('') +
    '</ol><div class="roadmap-reference"><span>遇到数学概念时</span>' + links(guides.slice(-1)) + '</div></section>';
}

function showHome() {
  const cards = guides.map((guide) => '<a class="chapter-card" href="#' + guide.id + '">' +
    '<div class="card-top">' + icon(guide.icon) + '<span>' + chapterNumber(guide) + '</span></div>' +
    '<h3>' + escape(guide.title) + '</h3><p>' + escape(guide.summary) + '</p></a>').join('');
  main.innerHTML = '<div class="home"><section class="hero" aria-labelledby="home-title">' +
    '<div class="hero-copy"><h1 id="home-title">算子基础知识</h1>' +
    '<p>面向算子开发新人的入门知识库。从理解模型计算，到编写算子、分析硬件与优化性能。</p>' +
    '<div class="hero-actions"><a class="button primary" href="#models">开始学习</a><a class="button secondary" href="#curriculum">浏览章节</a></div></div>' +
    showRoadmap() + '</section><section id="curriculum" class="curriculum" aria-labelledby="curriculum-title">' +
    '<header class="section-heading"><h2 id="curriculum-title">章节导览</h2><span>7 章 + 数学附录</span></header>' +
    '<div class="chapter-grid">' + cards + '</div></section>' +
    '<footer class="site-footer">算子研习册</footer></div>';
}

function showFlow(diagram) {
  const [title, steps] = diagram;
  return '<figure class="flow"><figcaption>' + escape(title) + '</figcaption><ol>' + steps.map(([name, description]) =>
    '<li><strong>' + escape(name) + '</strong><span>' + escape(description) + '</span></li>').join('') + '</ol></figure>';
}

function showTable(guide) {
  const [heading, label, rows] = guide.table;
  return '<div class="table-scroll" tabindex="0"><table><thead><tr><th scope="col">' + escape(heading) + '</th><th scope="col">' + escape(label) +
    '</th></tr></thead><tbody>' + rows.map(([name, detail]) => '<tr><th scope="row">' + escape(name) + '</th><td>' + escape(detail) + '</td></tr>').join('') +
    '</tbody></table></div>';
}

function topicList(index, sections) {
  const { id } = guides[index];
  return '<ul class="topic-list' + (sections.length >= 6 ? ' topic-list-wide' : '') + '">' + sections.map((section) => {
    const count = contents[index].children.get(section.id)?.length ?? 0;
    return '<li><a href="' + chapterHref(id, section.id) + '"><strong>' + escape(section.title) + '</strong>' +
      (count ? '<small>' + count + ' 个小节</small>' : '') + '</a></li>';
  }).join('') + '</ul>';
}

function neighbor(href, title, label) {
  return href ? '<a href="' + href + '"><small>' + label + '</small><strong>' + escape(title) + '</strong></a>' : '<span></span>';
}

function articleLayout(body, toc) {
  main.innerHTML = '<div class="reader"><article class="article">' + body +
    '</article>' + (toc ? '<aside class="article-toc"><nav aria-label="本页目录"><p>本页目录</p><ul>' +
      toc + '</ul></nav></aside>' : '') + '</div>';
}

function tocLink(id, target, label) {
  return '<li><a href="' + chapterHref(id, target) + '">' + escape(label) + '</a></li>';
}

function breadcrumb(items, label) {
  return '<nav class="breadcrumb" aria-label="当前位置"><ol>' + items.map(([href, title]) =>
    '<li><a href="' + href + '">' + escape(title) + '</a></li>').join('') +
    '<li aria-current="page">' + escape(label) + '</li></ol></nav>';
}

function showChapter(index) {
  const guide = guides[index];
  const entry = book[index];
  const topics = contents[index].children.get(null) ?? [];
  const note = guide.id === 'triton' ? '原稿教程 + 入门导读' :
    entry.sections.length ? '原稿提纲 + 入门导读' : '入门导读 · 原稿仅列章名';
  const prev = guides[index - 1];
  const next = guides[index + 1];
  articleLayout(breadcrumb([['#curriculum', '章节导览']], chapterNumber(guide)) +
    '<header class="article-heading"><h1>' + escape(guide.title) + '</h1><p>' + escape(guide.intro) + '</p><span class="source-note">' + note + '</span></header>' +
    (topics.length ? '<section id="full-chapter"><h2>本章小节</h2>' + topicList(index, topics) + '</section>' : '') +
    '<section id="guide"><h2>入门图解</h2>' + (guide.diagram ? showFlow(guide.diagram) : attentionDemo()) + showTable(guide) + '</section>' +
    (entry.html ? '<section class="prose">' + entry.html + '</section>' : '') +
    '<section id="exercise"><h2>动手想一想</h2><details class="exercise"><summary>' + escape(guide.exercise[0]) +
    '</summary><p>' + escape(guide.exercise[1]) + '</p></details></section>' +
    (guide.links ? '<section id="reading"><h2>延伸阅读</h2><ul class="reading">' + guide.links.map(([label, url]) =>
      '<li><a href="' + escape(url) + '">' + escape(label) + '</a></li>').join('') + '</ul></section>' : '') +
    '<nav class="pagination" aria-label="相邻章节">' +
      neighbor(prev && chapterHref(prev.id), prev?.title, '上一章') + neighbor(next && chapterHref(next.id), next?.title, '下一章') +
    '</nav>',
    (topics.length ? tocLink(guide.id, 'full-chapter', '本章小节') : '') +
    tocLink(guide.id, 'guide', '入门图解') + tocLink(guide.id, 'exercise', '动手想一想') +
    (guide.links ? tocLink(guide.id, 'reading', '延伸阅读') : ''));
}

function showTopic(index, page) {
  const guide = guides[index];
  const { children } = contents[index];
  const siblings = children.get(page.parentId);
  const position = siblings.indexOf(page);
  const prev = siblings[position - 1];
  const next = siblings[position + 1];
  const trail = sectionTrail(index, page.id).slice(0, -1);
  const subsectionList = children.get(page.id) ?? [];
  const linkedPages = subsectionList.filter((section) => section.isPage);
  const anchors = [];

  function renderSubsections(sections, level) {
    return sections.filter((section) => !section.isPage).map((section) => {
      anchors.push(section);
      const nested = children.get(section.id) ?? [];
      return '<h' + level + ' id="' + section.id + '">' + escape(section.title) + '</h' + level + '>' +
        section.html + renderSubsections(nested, level + 1);
    }).join('');
  }

  const lead = guide.topics?.[page.id];
  const subsections = renderSubsections(subsectionList, 2);
  const allOutline = anchors.length && anchors.every((section) => !section.html);
  const status = allOutline ? '<p class="section-status">以下小节为原稿提纲。</p>' : '';
  const prose = (lead ? '<p>' + escape(lead) + '</p>' : '') + page.html + status + subsections;
  const body = '<header class="article-heading topic-heading"><h1>' + escape(page.title) + '</h1></header>' +
    (prose ? '<section class="prose' + (allOutline ? ' outline-prose' : '') + '">' + prose + '</section>' : '') +
    (linkedPages.length ? '<section id="topic-list"><h2>小节目录</h2>' + topicList(index, linkedPages) + '</section>' : '');
  articleLayout(breadcrumb([['#curriculum', '章节导览'], [chapterHref(guide.id), guide.title],
    ...trail.map((section) => [chapterHref(guide.id, section.id), section.title])], page.id.slice(2)) + body +
    '<nav class="pagination" aria-label="相邻小节">' +
    neighbor(prev && chapterHref(guide.id, prev.id), prev?.title, '上一节') +
    neighbor(next && chapterHref(guide.id, next.id), next?.title, '下一节') + '</nav>',
    anchors.filter((section) => section.parentId === page.id).map((section) => tocLink(guide.id, section.id, section.title)).join(''));
}

function renderRoute(initial = false) {
  const previous = route;
  route = readRoute();
  const changedPage = !previous || previous.index !== route.index || previous.page !== route.page;
  setNav(false);
  showSidebar();
  if (changedPage) {
    if (route.index < 0) showHome();
    else if (route.page) showTopic(route.index, route.page);
    else showChapter(route.index);
    main.querySelectorAll('.attention').forEach((figure) => updateAttention(figure, 3));
  }
  document.title = route.index < 0 ? '算子研习册 · 算子基础知识' :
    (route.page?.title ?? guides[route.index].title) + ' · 算子研习册';
  const target = document.getElementById(route.id === 'curriculum' ? 'curriculum' : route.section || '');
  if (target) {
    target.tabIndex = -1;
    if (!initial) target.focus({ preventScroll: true });
    target.scrollIntoView();
  } else {
    if (!initial) main.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
}

const searchIndex = guides.flatMap((guide, index) => {
  const entry = book[index];
  return [
    { label: guide.title, href: '#' + guide.id, hint: guide.summary, text: guide.intro + ' ' + JSON.stringify(guide.table) },
    ...entry.sections.map((section) => ({
      label: section.title, href: chapterHref(guide.id, section.id), hint: guide.title,
      text: [section.searchText, guide.topics?.[section.id]].filter(Boolean).join(' '),
    })),
  ];
});

function renderSearch() {
  const query = searchInput.value.toLowerCase().trim();
  const words = query.split(/\s+/).filter(Boolean);
  const matches = words.length ? searchIndex.map((item) => {
    const label = item.label.toLowerCase();
    const text = item.text.toLowerCase();
    const score = words.every((word) => label.includes(word) || text.includes(word)) ?
      1 + words.filter((word) => label.includes(word)).length : 0;
    return { item, score };
  }).filter(({ score }) => score > 0).sort((a, b) => b.score - a.score).slice(0, 12).map(({ item }) => item) : [searchIndex[0],
    searchIndex.find((item) => item.href === '#transformer'), searchIndex.find((item) => item.href === '#triton')];
  searchResults.innerHTML = matches.length ? '<p class="results-label">' + (words.length ? '匹配结果' : '快速跳转') +
    '</p><ul>' + matches.map((item) => '<li><a href="' + item.href + '"><strong>' + escape(item.label) +
      '</strong><small>' + escape(item.hint) + '</small></a></li>').join('') + '</ul>' :
    '<p class="results-label">没有匹配结果。试试更短的关键词。</p>';
}

function openSearch() {
  if (dialog.open) return;
  searchOpener = document.activeElement;
  searchInput.value = '';
  renderSearch();
  dialog.showModal();
  searchInput.focus();
}

navToggle.addEventListener('click', () => setNav(!document.body.classList.contains('nav-open')));
scrim.addEventListener('click', () => { setNav(false); navToggle.focus(); });
mobileNav.addEventListener('change', () => setNav(false));
sidebar.addEventListener('click', (event) => {
  const toggle = event.target.closest('.tree-toggle');
  if (toggle) {
    const id = toggle.getAttribute('aria-controls');
    const open = toggle.getAttribute('aria-expanded') === 'false';
    toggle.setAttribute('aria-expanded', open);
    document.getElementById(id).hidden = !open;
    if (open) expandedBranches.add(id);
    else expandedBranches.delete(id);
  }
  if (event.target.closest('a')?.hash === location.hash) setNav(false);
});
document.querySelector('[data-search]').addEventListener('click', openSearch);
document.querySelector('#close-search').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => searchOpener?.focus());
searchInput.addEventListener('input', renderSearch);
searchInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') searchResults.querySelector('a')?.click();
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    searchResults.querySelector('a')?.focus();
  }
});
searchResults.addEventListener('click', (event) => {
  const link = event.target.closest('a');
  if (link) {
    dialog.close();
    if (link.hash === location.hash) renderRoute();
  }
});
document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    openSearch();
  }
  if (event.key === 'Escape' && document.body.classList.contains('nav-open')) {
    setNav(false);
    navToggle.focus();
  }
});
document.querySelector('.skip').addEventListener('click', (event) => {
  event.preventDefault();
  main.focus();
});
main.addEventListener('click', async (event) => {
  const token = event.target.closest('[data-token]');
  if (token) updateAttention(token.closest('.attention'), Number(token.dataset.token));
  const button = event.target.closest('[data-copy]');
  if (!button) return;
  const code = button.closest('.code-block').querySelector('code');
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.textContent = '已复制';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = '已选中文本';
  }
  setTimeout(() => { button.textContent = '复制'; }, 1600);
});
window.addEventListener('hashchange', () => renderRoute());
renderRoute(true);
