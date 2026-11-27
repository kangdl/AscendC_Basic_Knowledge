import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';

const prefix = '/AscendC_Basic_Knowledge/';
const files = new Map([
  ['index.html', 'text/html'],
  ['styles.css', 'text/css'],
  ['app.js', 'text/javascript'],
  ['content.json', 'application/json'],
]);

const server = createServer(async (request, response) => {
  const { pathname } = new URL(request.url, 'http://localhost');
  if (pathname === '/' || pathname === prefix.slice(0, -1)) {
    response.writeHead(302, { Location: prefix }).end();
    return;
  }
  const file = pathname.startsWith(prefix) ? pathname.slice(prefix.length) || 'index.html' : '';
  const type = files.get(file);
  if (!type) {
    response.writeHead(404).end('Not found');
    return;
  }
  try {
    const body = await readFile(new URL('./dist/' + file, import.meta.url));
    response.writeHead(200, { 'Content-Type': type + '; charset=utf-8' }).end(body);
  } catch {
    response.writeHead(500).end('Run node build.mjs before previewing.');
  }
});

server.listen(8765, '127.0.0.1', () => console.log('http://127.0.0.1:8765' + prefix));
