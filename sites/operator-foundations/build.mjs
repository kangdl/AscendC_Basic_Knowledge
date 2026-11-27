import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const path = (relative) => fileURLToPath(new URL(relative, import.meta.url));
const source = await readFile(path('../../ops_learn_index.md'), 'utf8');
const volume = source.match(/^# 第二卷 算子基础知识[ \t]*\r?$/m);
// Top-level headings in fenced examples are comments. Parse the volume line by line.
if (!volume) throw new Error('未找到第二卷 算子基础知识');
const headings = [];
const pages = [];
let current;
let section;
let ancestors = [];
let fenced = false;
for (const line of source.slice(volume.index).split(/\r?\n/).slice(1)) {
  if (/^```/.test(line)) fenced = !fenced;
  const heading = !fenced && line.match(/^(#{1,5})\s+(.+)$/);
  if (heading?.[1] === '#') break;
  if (heading?.[1] === '##') {
    current = { title: heading[2], sections: [], lines: [] };
    pages.push(current);
    section = undefined;
    ancestors = [];
  } else if (heading && current) {
    const title = heading[2].replace(/<a id="[^"]+"><\/a>/g, '').replace(/`/g, '').trim();
    const number = title.match(/^\d+(?:\.\d+)*/)?.[0];
    const level = heading[1].length;
    while (ancestors.at(-1)?.level >= level) ancestors.pop();
    section = {
      id: 's-' + number,
      title,
      level,
      parentId: ancestors.at(-1)?.id ?? null,
      lines: [],
    };
    ancestors.push(section);
    current.sections.push(section);
    headings.push(section);
  } else if (current) {
    (section ?? current).lines.push(line);
  }
}
if (fenced || pages.length !== 8 || headings.some(({ id }) => id === 's-undefined')) {
  throw new Error('第二卷目录与预期结构不符');
}

const escape = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

function inline(line) {
  const tokens = /(`[^`]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\)|https?:\/\/[^\s<，。；]+|\*\*[^*]+\*\*)/g;
  return line.split(tokens).map((part) => {
    if (part.startsWith('`') && part.endsWith('`')) return '<code>' + escape(part.slice(1, -1)) + '</code>';
    if (part.startsWith('**') && part.endsWith('**')) return '<strong>' + inline(part.slice(2, -2)) + '</strong>';
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (link) return '<a href="' + escape(link[2]) + '">' + escape(link[1]) + '</a>';
    if (/^https?:\/\/\S+$/.test(part)) return '<a href="' + escape(part) + '">' + escape(part) + '</a>';
    return escape(part);
  }).join('');
}

function table(rows) {
  const cells = (line, tag) => '<tr>' + line.trim().slice(1, -1)
    .split(/(?<!\\)\|/).map((cell) => '<' + tag + (tag === 'th' ? ' scope="col"' : '') +
      '>' + inline(cell.trim().replaceAll('\\|', '|')) + '</' + tag + '>').join('') + '</tr>';
  return '<div class="table-scroll" tabindex="0"><table><thead>' + cells(rows[0], 'th') +
    '</thead><tbody>' + rows.slice(2).map((row) => cells(row, 'td')).join('') + '</tbody></table></div>';
}

function blockKind(line) {
  if (!line.trim()) return 'blank';
  if (/^```/.test(line)) return 'code';
  if (/^\s*\|.*\|\s*$/.test(line)) return 'table';
  if (/^\s*[-*] /.test(line)) return 'ul';
  if (/^\s*\d+\. /.test(line)) return 'ol';
  if (/^\s*>/.test(line)) return 'quote';
  if (line.trim() === '---') return 'rule';
  return 'paragraph';
}

function list(lines, start) {
  const first = lines[start];
  const kind = blockKind(first);
  const indentOf = (line) => line.length - line.trimStart().length;
  const indent = indentOf(first);
  const items = [];
  let index = start;
  while (index < lines.length && blockKind(lines[index]) === kind && indentOf(lines[index]) === indent) {
    const item = lines[index++].trimStart().replace(/^(?:[-*]|\d+\.) /, '');
    let children = '';
    while (index < lines.length && ['ul', 'ol'].includes(blockKind(lines[index])) && indentOf(lines[index]) > indent) {
      const nested = list(lines, index);
      children += nested.html;
      index = nested.end;
    }
    items.push('<li>' + inline(item) + children + '</li>');
  }
  return { html: '<' + kind + '>' + items.join('') + '</' + kind + '>', end: index };
}

function render(lines) {
  const output = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index++];
    const kind = blockKind(line);
    if (kind === 'blank') continue;
    if (kind === 'code') {
      const code = [];
      while (index < lines.length && !/^```/.test(lines[index])) code.push(lines[index++]);
      index++;
      output.push('<div class="code-block"><div class="code-toolbar"><span>' + escape(line.slice(3).trim() || '示例') + '</span><button type="button" class="text-button" data-copy>复制</button></div><pre tabindex="0"><code>' + escape(code.join('\n')) + '</code></pre></div>');
      continue;
    }
    if (kind === 'rule') continue;
    if (kind === 'ul' || kind === 'ol') {
      const result = list(lines, index - 1);
      output.push(result.html);
      index = result.end;
      continue;
    }
    const group = [line];
    while (index < lines.length && blockKind(lines[index]) === kind) group.push(lines[index++]);
    if (kind === 'table') output.push(table(group));
    else if (kind === 'quote') output.push('<blockquote>' + group.map((item) => inline(item.replace(/^\s*>\s?/, ''))).join('<br>') + '</blockquote>');
    else output.push('<p>' + inline(group.join(' ')) + '</p>');
  }
  return output.join('\n');
}

await mkdir(path('./dist'), { recursive: true });
await Promise.all(['index.html', 'app.js', 'guides.js', 'styles.css'].map((file) => copyFile(path('./src/' + file), path('./dist/' + file))));
const book = pages.map(({ title, sections, lines }) => {
  const chapters = sections.map(({ lines: body, ...entry }) => ({
    ...entry,
    searchText: body.join(' ').replace(/\s+/g, ' ').trim(),
    html: render(body),
  }));
  const sizes = new Map(chapters.map((entry) => [entry.id, entry.searchText.length]));
  for (const entry of chapters.toReversed()) {
    if (entry.parentId) sizes.set(entry.parentId, sizes.get(entry.parentId) + sizes.get(entry.id));
  }
  return {
    title,
    sections: chapters.map((entry) => ({
      ...entry,
      // Keep short topics together. Give the children of long topics their own reading pages.
      isPage: entry.level === 3 || (entry.level === 4 && sizes.get(entry.parentId) > 6000),
    })),
    html: render(lines),
  };
});
await writeFile(path('./dist/content.js'), 'window.operatorBook = ' + JSON.stringify(book) + ';\n');
console.log('已生成 ' + pages.length + ' 个章节入口和 ' + headings.length + ' 个小节。可打开 dist/index.html。');
