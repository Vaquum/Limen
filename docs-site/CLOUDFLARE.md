# Cloudflare Deployment

This docs app is deployed as a Cloudflare Worker with static assets.

Public URL:

- `https://docs.vaquum.fi/limen/`

The Worker owns the hostname `docs.vaquum.fi` and serves the Docusaurus build from the `build/` directory through the `ASSETS` binding.

## Repo-side configuration

The required deploy configuration lives in this directory:

- [wrangler.jsonc](wrangler.jsonc)
- [src/worker.js](src/worker.js)

Important defaults:

- site URL defaults to `https://docs.vaquum.fi`
- base path defaults to `/limen/`

That means no Cloudflare environment variables are required for the standard Limen deployment.

## Cloudflare setup

In the current unified Cloudflare UI:

1. Go to `Workers & Pages`
2. Click `Create`
3. Choose the Git-based Worker deployment flow
4. Connect the `Vaquum/Limen` repository
5. Set the root directory to `docs-site`
6. Set the build command to `npm ci && npm run build`
7. Leave the deploy command as `npx wrangler deploy`
8. Deploy

After the Worker is created:

1. Open the Worker
2. Go to `Settings`
3. Open `Domains & Routes`
4. Add the custom domain `docs.vaquum.fi`

## Runtime behavior

The Worker does three things:

- redirects `/` to `/limen/`
- redirects `/limen` to `/limen/`
- serves `/limen/*` from the Docusaurus static build

Examples:

- `https://docs.vaquum.fi/` -> redirect to `https://docs.vaquum.fi/limen/`
- `https://docs.vaquum.fi/limen/guides/advanced-search` -> serves the `Advanced Search` page

## Local build check

From this directory:

```bash
npm ci
npm run build
```

This writes the static output to `build/`.
