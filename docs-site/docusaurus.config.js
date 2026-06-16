const path = require('path');
const {themes: prismThemes} = require('prism-react-renderer');
const rehypeKatex = require('rehype-katex').default;
const remarkMath = require('remark-math').default;
const productDocs = require('./product-docs.json');

function normalizeBaseUrl(value) {
  if (!value || value === '/') {
    return '/';
  }

  const withLeadingSlash = value.startsWith('/') ? value : `/${value}`;
  return withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`;
}

const baseUrl = normalizeBaseUrl(process.env.DOCS_BASE_URL || productDocs.basePath);
const url = process.env.DOCS_SITE_URL || 'https://docs.vaquum.fi';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: productDocs.productName,
  tagline: productDocs.tagline,
  url,
  baseUrl,
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  favicon: 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>L</text></svg>',
  trailingSlash: false,
  organizationName: 'Vaquum',
  projectName: 'Limen',
  themes: [],
  stylesheets: ['https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css'],
  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        docsRouteBasePath: '/',
        indexDocs: true,
        indexBlog: false,
        hashed: true,
      },
    ],
  ],
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: path.resolve(__dirname, '.generated/docs'),
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: `${productDocs.sourceRepoUrl}/tree/main/`,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        pages: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      metadata: [
        {name: 'description', content: productDocs.tagline},
        {property: 'og:type', content: 'website'},
        {property: 'og:site_name', content: productDocs.productName},
        {property: 'og:title', content: `${productDocs.productName} Docs`},
        {property: 'og:description', content: productDocs.tagline},
        {property: 'og:url', content: `${url}${baseUrl}`},
        {name: 'twitter:card', content: 'summary'},
        {name: 'twitter:title', content: `${productDocs.productName} Docs`},
        {name: 'twitter:description', content: productDocs.tagline},
      ],
      navbar: {
        title: productDocs.productName,
        items: [
          {to: '/', label: 'Home', position: 'left'},
          {to: '/overview', label: 'Overview', position: 'left'},
          {to: '/guides', label: 'Guides', position: 'left'},
          {to: '/reference', label: 'Reference', position: 'left'},
          {to: '/developer', label: 'Developer', position: 'left'},
          {to: '/packages', label: 'Packages', position: 'left'},
          {
            href: productDocs.sourceRepoUrl,
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Overview', to: '/overview'},
              {label: 'Guides', to: '/guides'},
              {label: 'Reference', to: '/reference'},
              {label: 'Developer', to: '/developer'},
              {label: 'Packages', to: '/packages'},
            ],
          },
          {
            title: 'Product',
            items: [
              {label: 'Limen Repository', href: productDocs.sourceRepoUrl},
              {label: 'Vaquum', href: 'https://github.com/Vaquum'},
            ],
          },
        ],
        copyright: `Copyright ${new Date().getFullYear()} Vaquum.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      docs: {
        sidebar: {
          hideable: true,
        },
      },
    }),
};

module.exports = config;
