/* 用最小 DOM 替身跑一遍报告里的前端脚本，把运行时错误在生成阶段就暴露出来。
   这两份 HTML 是离线交付物，没有构建流程也没有浏览器可用；如果 JS 里有一个拼错的
   字段名，页面会静默留下空白区块，而生成脚本本身完全不会报错。所以这里把 script
   抽出来在 node 里执行，并检查每个预期挂载点是否真的被写入了内容。 */

import { readFileSync } from "node:fs";

const file = process.argv[2];
const expected = process.argv.slice(3);
const html = readFileSync(file, "utf8");

const jsonBlobs = {};
for (const m of html.matchAll(
  /<script id="([^"]+)" type="application\/json">([\s\S]*?)<\/script>/g
)) {
  jsonBlobs[m[1]] = m[2];
}
const codeMatch = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if (!codeMatch.length) throw new Error("找不到可执行的 script 块");
const code = codeMatch[codeMatch.length - 1][1];

const written = new Set();
function makeEl(id) {
  return {
    id,
    dataset: {},
    style: {},
    value: "",
    checked: false,
    classList: { add() {}, remove() {}, toggle() {} },
    textContent: jsonBlobs[id] ?? "",
    set innerHTML(v) {
      if (v && String(v).length) written.add(id);
      this._html = v;
    },
    get innerHTML() {
      return this._html ?? "";
    },
    appendChild() {},
    replaceChildren() {},
    addEventListener() {},
    getBoundingClientRect: () => ({ top: 0 }),
    querySelectorAll: () => [],
  };
}
const els = new Map();
globalThis.document = {
  getElementById(id) {
    if (!els.has(id)) els.set(id, makeEl(id));
    return els.get(id);
  },
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: (t) => makeEl("new:" + t),
  addEventListener() {},
};
globalThis.window = globalThis;
globalThis.localStorage = { getItem() { return null; }, setItem() {} };

try {
  new Function(code)();
} catch (e) {
  console.error(`✗ ${file} 运行时报错：${e.stack}`);
  process.exit(1);
}

const missing = expected.filter((id) => !written.has(id));
if (missing.length) {
  console.error(`✗ ${file} 以下区块没有被写入内容：${missing.join(", ")}`);
  process.exit(1);
}
console.log(`✓ ${file} 脚本执行通过，${written.size} 个区块已渲染`);
