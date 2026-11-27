import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { publicFileType } from './resource-path.mjs';

const prefix = '/AscendC_Basic_Knowledge/';
const port = Number(process.argv[2] ?? 8765);
const dist = fileURLToPath(new URL('./dist/', import.meta.url));

const server = createServer(async (request, response) => {
  const { pathname } = new URL(request.url, 'http://localhost');
  if (pathname === '/' || pathname === prefix.slice(0, -1)) {
    response.writeHead(302, { Location: prefix }).end();
    return;
  }
  let file = '';
  try {
    if (pathname.startsWith(prefix) && !/%2f/i.test(pathname)) file = decodeURIComponent(pathname.slice(prefix.length)) || 'index.html';
  } catch {
    response.writeHead(400).end('Invalid URL');
    return;
  }
  const type = publicFileType(file);
  if (!type) {
    response.writeHead(404).end('Not found');
    return;
  }
  try {
    const body = await readFile(join(dist, file));
    response.writeHead(200, { 'Content-Type': type + (type.startsWith('text/') ? '; charset=utf-8' : ''), 'Cache-Control': 'no-store' }).end(body);
  } catch (error) {
    if (error.code === 'ENOENT') {
      response.writeHead(404).end('Not found');
    } else {
      response.writeHead(500).end('Run node build.mjs before previewing.');
    }
  }
});

server.listen(port, '127.0.0.1', () => console.log('http://127.0.0.1:' + server.address().port + prefix));
