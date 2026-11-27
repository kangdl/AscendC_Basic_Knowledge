import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const library = fileURLToPath(new URL('./node_modules/mermaid/dist/mermaid.min.js', import.meta.url));

export const narrowSource = (source) => source.replace(/^(flowchart|graph)[ \t]+LR\b/m, '$1 TB');

export async function renderDiagrams(diagrams, stylesheet) {
  if (!diagrams.size) return new Map();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ offline: true });
    await page.setContent('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body></body></html>');
    await page.addStyleTag({ content: stylesheet });
    await page.addScriptTag({ path: library });
    await page.evaluate(async () => {
      await document.fonts.ready;
      const style = getComputedStyle(document.documentElement);
      const token = (name) => style.getPropertyValue('--' + name).trim();
      window.mermaid.initialize({
        startOnLoad: false, securityLevel: 'strict', htmlLabels: false, theme: 'base',
        flowchart: { nodeSpacing: 24, rankSpacing: 42, padding: 10 },
        suppressErrorRendering: true,
        themeVariables: {
          fontFamily: token('sans'), primaryColor: token('rail'), primaryTextColor: token('ink'),
          primaryBorderColor: token('rule'), lineColor: token('muted'), edgeLabelBackground: token('card'),
        },
        themeCSS: '.edgeLabel text { paint-order: stroke; stroke: ' + token('card') + '; stroke-width: 4px; stroke-linejoin: round; }',
      });
    });
    const files = new Map();
    for (const [id, { source, caption }] of diagrams) {
      try {
        const narrow = narrowSource(source);
        const variants = narrow === source ? [[id, source]] : [[id, source], [id + '-narrow', narrow]];
        for (const [variantId, definition] of variants) {
          const svg = await page.evaluate(async ({ variantId, definition }) =>
            (await window.mermaid.render('mermaid-' + variantId, definition)).svg, { variantId, definition });
          files.set('diagrams/' + variantId + '.svg', svg + '\n');
        }
      } catch (error) {
        throw new Error('Mermaid 生成失败：' + (caption ?? source.split('\n')[0]) + '\n' + error.message, { cause: error });
      }
    }
    return files;
  } finally {
    await browser.close();
  }
}
