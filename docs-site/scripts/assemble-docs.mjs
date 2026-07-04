import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), '..', '..');
const siteRoot = path.resolve(repoRoot, 'docs-site');
const outRoot = path.resolve(siteRoot, '.generated', 'docs');

const repoBlobBaseUrl = 'https://github.com/Vaquum/Limen/blob/main';
const repoTreeBaseUrl = 'https://github.com/Vaquum/Limen/tree/main';

const sectionCategories = [
  {
    dir: 'overview',
    label: 'Overview',
    position: 1,
    slug: '/overview',
    description: 'What Limen is, how it fits together, and where to start.',
  },
  {
    dir: 'guides',
    label: 'Guides',
    position: 2,
    slug: '/guides',
    description: 'End-to-end workflows for running research in Limen.',
  },
  {
    dir: 'reference',
    label: 'Reference',
    position: 3,
    slug: '/reference',
    description: 'Detailed interface and library reference for Limen.',
  },
  {
    dir: 'developer',
    label: 'Developer',
    position: 4,
    slug: '/developer',
    description: 'Contributor, release, and documentation maintenance guides.',
  },
  {
    dir: 'packages',
    label: 'Packages',
    position: 5,
    slug: '/packages',
    description: 'Module ownership, boundaries, and canonical entry points.',
  },
];

const docs = [
  {
    source: 'README.md',
    dest: 'index.md',
    slug: '/',
    title: 'Limen',
    sidebarLabel: 'Home',
  },
  {
    source: 'docs/README.md',
    dest: 'overview/docs-hub.md',
    slug: '/overview/docs-hub',
    sidebarPosition: 1,
  },
  {
    source: 'docs/Historical-Data.md',
    dest: 'guides/historical-data.md',
    slug: '/guides/historical-data',
    sidebarPosition: 1,
  },
  {
    source: 'docs/End-to-End-Workflow.md',
    dest: 'guides/end-to-end-workflow.md',
    slug: '/guides/end-to-end-workflow',
    sidebarPosition: 2,
  },
  {
    source: 'docs/Data-Bars.md',
    dest: 'guides/data-bars.md',
    slug: '/guides/data-bars',
    sidebarPosition: 3,
  },
  {
    source: 'docs/Single-File-Decoder.md',
    dest: 'guides/single-file-decoder.md',
    slug: '/guides/single-file-decoder',
    sidebarPosition: 4,
  },
  {
    source: 'docs/Built-In-SFDs.md',
    dest: 'guides/built-in-sfds.md',
    slug: '/guides/built-in-sfds',
    sidebarPosition: 5,
  },
  {
    source: 'docs/Experiment-Manifest.md',
    dest: 'guides/experiment-manifest.md',
    slug: '/guides/experiment-manifest',
    sidebarPosition: 6,
  },
  {
    source: 'docs/Universal-Experiment-Loop.md',
    dest: 'guides/universal-experiment-loop.md',
    slug: '/guides/universal-experiment-loop',
    sidebarPosition: 7,
  },
  {
    source: 'docs/Advanced-Search.md',
    dest: 'guides/advanced-search.md',
    slug: '/guides/advanced-search',
    sidebarPosition: 8,
  },
  {
    source: 'docs/Reducers-And-Feedback.md',
    dest: 'guides/reducers-and-feedback.md',
    slug: '/guides/reducers-and-feedback',
    sidebarPosition: 9,
  },
  {
    source: 'docs/Log.md',
    dest: 'guides/log.md',
    slug: '/guides/log',
    sidebarPosition: 10,
  },
  {
    source: 'docs/Benchmark.md',
    dest: 'guides/benchmark.md',
    slug: '/guides/benchmark',
    sidebarPosition: 11,
  },
  {
    source: 'docs/Backtest.md',
    dest: 'guides/backtest.md',
    slug: '/guides/backtest',
    sidebarPosition: 12,
  },
  {
    source: 'docs/Trainer.md',
    dest: 'guides/trainer.md',
    slug: '/guides/trainer',
    sidebarPosition: 13,
  },
  {
    source: 'docs/Cohort.md',
    dest: 'guides/cohort.md',
    slug: '/guides/cohort',
    sidebarPosition: 14,
  },
  {
    source: 'docs/Conserved-Flux-Renormalization.md',
    dest: 'guides/conserved-flux-renormalization.md',
    slug: '/guides/conserved-flux-renormalization',
    sidebarPosition: 15,
  },
  {
    source: 'docs/Glossary.md',
    dest: 'reference/glossary.md',
    slug: '/reference/glossary',
    sidebarPosition: 1,
  },
  {
    source: 'docs/Indicators.md',
    dest: 'reference/indicators.md',
    slug: '/reference/indicators',
    sidebarPosition: 2,
  },
  {
    source: 'docs/Features.md',
    dest: 'reference/features.md',
    slug: '/reference/features',
    sidebarPosition: 3,
  },
  {
    source: 'docs/Targets.md',
    dest: 'reference/targets.md',
    slug: '/reference/targets',
    sidebarPosition: 4,
  },
  {
    source: 'docs/Transforms.md',
    dest: 'reference/transforms.md',
    slug: '/reference/transforms',
    sidebarPosition: 5,
  },
  {
    source: 'docs/Scalers.md',
    dest: 'reference/scalers.md',
    slug: '/reference/scalers',
    sidebarPosition: 6,
  },
  {
    source: 'docs/Calibration.md',
    dest: 'reference/calibration.md',
    slug: '/reference/calibration',
    sidebarPosition: 7,
  },
  {
    source: 'docs/Standard-Metrics-Library.md',
    dest: 'reference/standard-metrics-library.md',
    slug: '/reference/standard-metrics-library',
    sidebarPosition: 8,
  },
  {
    source: 'docs/Reference-Architecture.md',
    dest: 'reference/reference-architecture.md',
    slug: '/reference/reference-architecture',
    sidebarPosition: 9,
  },
  {
    source: 'docs/Utilities.md',
    dest: 'reference/utilities.md',
    slug: '/reference/utilities',
    sidebarPosition: 10,
  },
  {
    source: 'docs/Command-Line-Interface.md',
    dest: 'reference/command-line-interface.md',
    slug: '/reference/command-line-interface',
    sidebarPosition: 11,
  },
  {
    source: 'docs/Developer/README.md',
    dest: 'developer/developer-home.md',
    slug: '/developer/home',
    sidebarPosition: 1,
    sidebarLabel: 'Developer Home',
  },
  {
    source: 'docs/Developer/Documentation-System.md',
    dest: 'developer/documentation-system.md',
    slug: '/developer/documentation-system',
    sidebarPosition: 2,
  },
  {
    source: 'docs/Developer/Pruning-Strategies.md',
    dest: 'developer/pruning-strategies.md',
    slug: '/developer/pruning-strategies',
    sidebarPosition: 3,
  },
  {
    source: 'docs/Developer/Writing-Docstrings.md',
    dest: 'developer/writing-docstrings.md',
    slug: '/developer/writing-docstrings',
    sidebarPosition: 4,
  },
  {
    source: 'docs/Developer/Contributing-Foundational-SFDs.md',
    dest: 'developer/contributing-foundational-sfds.md',
    slug: '/developer/contributing-foundational-sfds',
    sidebarPosition: 5,
  },
  {
    source: 'docs/Developer/Making-Release.md',
    dest: 'developer/making-release.md',
    slug: '/developer/making-release',
    sidebarPosition: 6,
  },
  {
    source: 'docs/Semantic-Versioning.md',
    dest: 'developer/semantic-versioning.md',
    slug: '/developer/semantic-versioning',
    sidebarPosition: 7,
  },
  {
    source: 'docs/TechnicalDebt.md',
    dest: 'developer/technical-debt.md',
    slug: '/developer/technical-debt',
    sidebarPosition: 8,
  },
  {
    source: 'docs/Audit-Closeout.md',
    dest: 'developer/audit-closeout.md',
    slug: '/developer/audit-closeout',
    sidebarPosition: 9,
  },
  {
    source: 'limen/data/README.md',
    dest: 'packages/data.md',
    slug: '/packages/data',
    sidebarPosition: 1,
    sidebarLabel: 'Data',
  },
  {
    source: 'limen/experiment/README.md',
    dest: 'packages/experiment.md',
    slug: '/packages/experiment',
    sidebarPosition: 2,
    sidebarLabel: 'Experiment',
  },
  {
    source: 'limen/sfd/README.md',
    dest: 'packages/sfd.md',
    slug: '/packages/sfd',
    sidebarPosition: 3,
    sidebarLabel: 'SFD',
  },
  {
    source: 'limen/indicators/README.md',
    dest: 'packages/indicators.md',
    slug: '/packages/indicators',
    sidebarPosition: 4,
    sidebarLabel: 'Indicators',
  },
  {
    source: 'limen/features/README.md',
    dest: 'packages/features.md',
    slug: '/packages/features',
    sidebarPosition: 5,
    sidebarLabel: 'Features',
  },
  {
    source: 'limen/transforms/README.md',
    dest: 'packages/transforms.md',
    slug: '/packages/transforms',
    sidebarPosition: 6,
    sidebarLabel: 'Transforms',
  },
  {
    source: 'limen/scalers/README.md',
    dest: 'packages/scalers.md',
    slug: '/packages/scalers',
    sidebarPosition: 7,
    sidebarLabel: 'Scalers',
  },
  {
    source: 'limen/metrics/README.md',
    dest: 'packages/metrics.md',
    slug: '/packages/metrics',
    sidebarPosition: 8,
    sidebarLabel: 'Metrics',
  },
  {
    source: 'limen/log/README.md',
    dest: 'packages/log.md',
    slug: '/packages/log',
    sidebarPosition: 9,
    sidebarLabel: 'Log',
  },
  {
    source: 'limen/cohort/README.md',
    dest: 'packages/cohort.md',
    slug: '/packages/cohort',
    sidebarPosition: 10,
    sidebarLabel: 'Cohort',
  },
  {
    source: 'limen/backtest/README.md',
    dest: 'packages/backtest.md',
    slug: '/packages/backtest',
    sidebarPosition: 11,
    sidebarLabel: 'Backtest',
  },
  {
    source: 'limen/utils/README.md',
    dest: 'packages/utils.md',
    slug: '/packages/utils',
    sidebarPosition: 12,
    sidebarLabel: 'Utils',
  },
  {
    source: 'limen/calibration/README.md',
    dest: 'packages/calibration.md',
    slug: '/packages/calibration',
    sidebarPosition: 13,
    sidebarLabel: 'Calibration',
  },
  {
    source: 'limen/cli/README.md',
    dest: 'packages/cli.md',
    slug: '/packages/cli',
    sidebarPosition: 14,
    sidebarLabel: 'CLI',
  },
  {
    source: 'limen/targets/README.md',
    dest: 'packages/targets.md',
    slug: '/packages/targets',
    sidebarPosition: 15,
    sidebarLabel: 'Targets',
  },
  {
    source: 'limen/yaml/README.md',
    dest: 'packages/yaml.md',
    slug: '/packages/yaml',
    sidebarPosition: 16,
    sidebarLabel: 'YAML',
  },
];

const mappingBySource = new Map(
  docs.map((doc) => [normalizePath(doc.source), doc])
);

function normalizePath(value) {
  return value.split(path.sep).join('/');
}

async function ensureDir(dir) {
  await fs.mkdir(dir, {recursive: true});
}

async function writeJson(filePath, value) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function buildFrontMatter(doc) {
  const lines = ['---'];

  if (doc.slug) {
    lines.push(`slug: ${doc.slug}`);
  }

  if (doc.title) {
    lines.push(`title: ${doc.title}`);
  }

  if (typeof doc.sidebarPosition === 'number') {
    lines.push(`sidebar_position: ${doc.sidebarPosition}`);
  }

  if (doc.sidebarLabel) {
    lines.push(`sidebar_label: ${doc.sidebarLabel}`);
  }

  if (doc.dest === 'index.md') {
    lines.push('pagination_next: null');
    lines.push('pagination_prev: null');
  }

  lines.push(`custom_edit_url: ${repoBlobBaseUrl}/${doc.source}`);
  lines.push('---', '');
  return lines.join('\n');
}

function resolveDocLink(fromSource, target) {
  if (!target || target.startsWith('http://') || target.startsWith('https://') || target.startsWith('mailto:') || target.startsWith('#')) {
    return target;
  }

  const [targetPath, targetHash] = target.split('#');

  if (!targetPath) {
    return target;
  }

  if (targetPath.startsWith('/')) {
    return target;
  }

  const resolvedSource = normalizePath(path.posix.normalize(path.posix.join(path.posix.dirname(normalizePath(fromSource)), targetPath)));
  let targetDoc = mappingBySource.get(resolvedSource);

  if (!targetDoc && !path.posix.extname(resolvedSource)) {
    targetDoc = mappingBySource.get(normalizePath(path.posix.join(resolvedSource, 'README.md')));
  }

  if (!targetDoc) {
    const repoFsPath = path.resolve(repoRoot, resolvedSource);

    if (fsSync.existsSync(repoFsPath)) {
      const repoUrlBase = fsSync.statSync(repoFsPath).isDirectory()
        ? repoTreeBaseUrl
        : repoBlobBaseUrl;
      return targetHash
        ? `${repoUrlBase}/${resolvedSource}#${targetHash}`
        : `${repoUrlBase}/${resolvedSource}`;
    }

    return target;
  }

  const currentDoc = mappingBySource.get(normalizePath(fromSource));
  const fromDest = currentDoc ? normalizePath(currentDoc.dest) : '';
  const toDest = normalizePath(targetDoc.dest);
  let relative = normalizePath(path.posix.relative(path.posix.dirname(fromDest), toDest));

  if (!relative) {
    relative = path.posix.basename(toDest);
  }

  return targetHash ? `${relative}#${targetHash}` : relative;
}

function rewriteLinks(content, fromSource) {
  return content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, target) => {
    const rewritten = resolveDocLink(fromSource, target.trim());
    return `[${label}](${rewritten})`;
  });
}

function rewriteOutsideCode(content, transform) {
  let out = '';
  let index = 0;
  let inFence = false;
  let inInlineCode = false;
  let plainStart = 0;

  while (index < content.length) {
    if (content.startsWith('```', index)) {
      if (plainStart < index) {
        out += transform(content.slice(plainStart, index));
      }
      inFence = !inFence;
      out += '```';
      index += 3;
      plainStart = index;
      continue;
    }

    if (!inFence && content[index] === '`') {
      if (plainStart < index) {
        out += transform(content.slice(plainStart, index));
      }
      inInlineCode = !inInlineCode;
      out += '`';
      index += 1;
      plainStart = index;
      continue;
    }

    index += 1;
  }

  if (plainStart < content.length) {
    out += transform(content.slice(plainStart));
  }

  return out;
}

function normalizeForMdx(content) {
  return rewriteOutsideCode(content, (chunk) =>
    chunk
      .replace(
        /<p align="center">([\s\S]*?)<\/p>/g,
        '<div align="center">$1</div>'
      )
      .replace(/<br>/g, '<br />')
      .replace(/<hr>/g, '<hr />')
      .replace(/<img([^>]*?)(?<!\/)>/g, '<img$1 />')
  );
}

function normalizeReferencePlaceholders(content, doc) {
  if (!doc.dest.startsWith('reference/')) {
    return content;
  }

  let out = '';
  let index = 0;
  let inFence = false;
  let inInlineCode = false;

  while (index < content.length) {
    if (content.startsWith('```', index)) {
      inFence = !inFence;
      out += '```';
      index += 3;
      continue;
    }

    if (!inFence && content[index] === '`') {
      inInlineCode = !inInlineCode;
      out += '`';
      index += 1;
      continue;
    }

    if (!inFence && !inInlineCode) {
      const match = content.slice(index).match(/^\{([a-zA-Z0-9_:+.-]+)\}/);
      if (match) {
        out += `(${match[1]})`;
        index += match[0].length;
        continue;
      }
    }

    out += content[index];
    index += 1;
  }

  return out;
}

async function copyDoc(doc) {
  const sourcePath = path.resolve(repoRoot, doc.source);
  const destPath = path.resolve(outRoot, doc.dest);
  const raw = await fs.readFile(sourcePath, 'utf8');
  const rewritten = normalizeReferencePlaceholders(
    normalizeForMdx(rewriteLinks(raw, doc.source)),
    doc
  );
  const output = `${buildFrontMatter(doc)}${rewritten}`;

  await ensureDir(path.dirname(destPath));
  await fs.writeFile(destPath, output);
}

async function writeCategoryFiles() {
  for (const category of sectionCategories) {
    const categoryPath = path.resolve(outRoot, category.dir, '_category_.json');
    await writeJson(categoryPath, {
      label: category.label,
      position: category.position,
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        slug: category.slug,
        title: category.label,
        description: category.description,
      },
    });
  }
}

async function main() {
  await fs.rm(outRoot, {recursive: true, force: true});
  await ensureDir(outRoot);
  await writeCategoryFiles();

  for (const doc of docs) {
    await copyDoc(doc);
  }
}

await main();
