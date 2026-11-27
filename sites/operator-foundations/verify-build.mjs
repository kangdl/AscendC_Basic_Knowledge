import { lstat, readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { publicFileType, siteFiles } from './resource-path.mjs';

export async function verifyBuild(directory) {
  const { files } = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8'));
  if (!Array.isArray(files) || files.some((file) => typeof file !== 'string' || !publicFileType(file))) {
    throw new Error('发布清单包含不支持的路径');
  }
  const expected = new Set(files);
  if (expected.size !== files.length || [...siteFiles.keys()].some((file) => !expected.has(file))) {
    throw new Error('发布清单有重复项目或缺少站点文件');
  }

  async function checkDirectory(prefix = '') {
    for (const entry of await readdir(join(directory, prefix), { withFileTypes: true })) {
      const file = prefix + entry.name;
      if (entry.isDirectory()) {
        if (!files.some((path) => path.startsWith(file + '/'))) throw new Error('发布产物包含未列出的目录：' + file);
        await checkDirectory(file + '/');
      } else if (!entry.isFile() || !expected.delete(file) || !(await lstat(join(directory, file))).size) {
        throw new Error('发布产物包含未列出、为空或不是普通文件的项目：' + file);
      }
    }
  }
  await checkDirectory();
  if (expected.size) throw new Error('发布产物缺少文件：' + [...expected].join(', '));
  JSON.parse(await readFile(join(directory, 'content.json'), 'utf8'));
  return files.length;
}

if (import.meta.main) {
  console.log('已校验 ' + await verifyBuild(fileURLToPath(new URL('./dist', import.meta.url))) + ' 个发布文件。');
}
