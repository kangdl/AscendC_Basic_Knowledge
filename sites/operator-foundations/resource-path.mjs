export const siteFiles = new Map([
  ['index.html', 'text/html'],
  ['styles.css', 'text/css'],
  ['app.js', 'text/javascript'],
  ['content.json', 'application/json'],
  ['manifest.json', 'application/json'],
]);

const imageTypes = new Map([
  ['avif', 'image/avif'], ['gif', 'image/gif'], ['ico', 'image/x-icon'],
  ['jpeg', 'image/jpeg'], ['jpg', 'image/jpeg'], ['png', 'image/png'], ['svg', 'image/svg+xml'], ['webp', 'image/webp'],
]);

export function resourceType(path) {
  const [root, ...segments] = path.split('/');
  if (!segments.length || !['resources', 'img', 'images'].includes(root) || segments.some((segment) =>
    !segment || segment.startsWith('.') || /[\\:%?#<>"|*\u0000-\u001f]|[ .]$/.test(segment))) return undefined;
  const extension = segments.at(-1).match(/\.([^.]+)$/)?.[1].toLowerCase();
  if (root === 'resources' && extension === 'html') return 'text/html';
  return imageTypes.get(extension);
}

export function publicFileType(path) {
  return siteFiles.get(path) ?? resourceType(path) ??
    (/^diagrams\/[a-f0-9]{24}(?:-narrow)?\.(?:svg|png)$/.test(path) ? imageTypes.get(path.split('.').at(-1)) : undefined);
}
