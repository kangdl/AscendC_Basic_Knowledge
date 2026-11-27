import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const library = fileURLToPath(new URL('./node_modules/mermaid/dist/mermaid.min.js', import.meta.url));

export const narrowSource = (source) => source.replace(/^(flowchart|graph)[ \t]+LR\b/m, '$1 TB');
export const diagramExtension = (source) => /\$\$[^\r\n]+?\$\$/.test(source) ? 'png' : 'svg';

export async function renderDiagrams(diagrams, stylesheet) {
  if (!diagrams.size) return new Map();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ offline: true, deviceScaleFactor: 2 });
    await page.setContent('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body></body></html>');
    await page.addStyleTag({ content: stylesheet });
    await page.addScriptTag({ path: library });
    const config = await page.evaluate(async () => {
      await document.fonts.ready;
      const style = getComputedStyle(document.documentElement);
      const token = (name) => style.getPropertyValue('--' + name).trim();
      return {
        startOnLoad: false, securityLevel: 'strict', theme: 'base',
        flowchart: { nodeSpacing: 24, rankSpacing: 42, padding: 10 },
        suppressErrorRendering: true,
        themeVariables: {
          fontFamily: token('sans'), primaryColor: token('rail'), primaryTextColor: token('ink'),
          primaryBorderColor: token('rule'), lineColor: token('muted'), edgeLabelBackground: token('card'),
        },
        themeCSS: '.edgeLabel text { paint-order: stroke; stroke: ' + token('card') + '; stroke-width: 4px; stroke-linejoin: round; }',
      };
    });
    const files = new Map();
    for (const [id, { source, caption }] of diagrams) {
      try {
        const extension = diagramExtension(source);
        const htmlLabels = extension === 'png';
        await page.evaluate((options) => window.mermaid.initialize(options), { ...config, htmlLabels });
        const narrow = narrowSource(source);
        const variants = narrow === source ? [[id, source]] : [[id, source], [id + '-narrow', narrow]];
        for (const [variantId, definition] of variants) {
          const svg = await page.evaluate(async ({ variantId, definition }) =>
            (await window.mermaid.render('mermaid-' + variantId, definition)).svg, { variantId, definition });
          const file = 'diagrams/' + variantId + '.' + extension;
          if (htmlLabels) {
            await page.evaluate(async (markup) => {
              document.body.innerHTML = markup;
              document.body.style.cssText = 'margin:0;background:transparent';
              const element = document.body.querySelector('svg');
              if (!element.querySelector('math') || element.textContent.includes('$$')) {
                throw new Error('图内 LaTeX 公式未完成排版');
              }
              const { width, height } = element.viewBox.baseVal;
              element.style.cssText += ';display:block;max-width:none;width:' + width + 'px;height:' + height + 'px';
              await document.fonts.ready;
            }, svg);
            files.set(file, await page.locator('body > svg').screenshot({ omitBackground: true }));
            await page.locator('body > svg').evaluate((element) => element.remove());
          } else {
            files.set(file, svg + '\n');
          }
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
