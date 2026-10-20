# Cloudflare deployment

The existing Docusaurus renderer still serves [docs.vaquum.fi/limen/](https://docs.vaquum.fi/limen/). The replacement research reader and release-bound publisher live in Limen. Their first production cutover remains an acceptance obligation; candidate execution does not establish deployment.

## Serving boundary

The `limen` Worker serves `/limen/*` through `ASSETS`. It redirects `/` and `/limen` to `/limen/`, and root robots/sitemap requests to their product resources. Other product prefixes return 404. Redirects, errors and assets retain HSTS, CSP, permissions, referrer, content-type and frame protections.

The Worker imports `src/security-headers.json`. The build retains that exact policy in its hashed site payload. Verification compares every response's six security-header values with that release's policy, including the 404 response. Rollback retains the previous release's policy.

The legacy [wrangler.jsonc](wrangler.jsonc) reads `build/`. The replacement [publisher/wrangler.json](publisher/wrangler.json) reads the sealed `reader-build/`. The replacement pins Wrangler and its complete dependency graph in `publisher/package-lock.json`; it requires Node.js 22 or later. Reader checks continue to run under Node.js 20 in the required Docs Site job.

## Release-bound publication

[docs_publish.yml](../.github/workflows/docs_publish.yml) follows a successful canonical package-publishing run. Its run-scoped identity artifact names the exact merged tag, commit and wheel digest. Missing identity, failed package publication, changed source or a substituted wheel prevents deployment.

The publisher downloads the actual public PyPI wheel and the immutable project/input bundle from the matching GitHub Release. It runs the unchanged projects in a fresh environment outside the checkout, verifies complete semantic and documentation evidence, builds the reader, and runs its browser matrix. Contributor and fork checks need no deployment secrets.

After proof, the publisher seals the build and retains its source, results, wheel, environment, validation and rollback artifacts on the public GitHub Release. Uploads never replace existing bytes. The site's `publication.json` selects a detached `publication-SHA256.json`; every earlier manifest remains available for verification and rollback. Wheel downloads retain their installable distribution filename.

`deploy_publication.py stage` uploads an inactive Worker version and verifies its anonymous Version URL against those public artifacts. `activate` repeats verification, confirms that independent Git builds are disconnected and that no newer package release superseded the build, then activates exactly that version. Failure restores and verifies the recorded predecessor. The first migration retains the existing legacy Worker version and home-page digest explicitly; later rollbacks also verify the predecessor's complete publication envelope.

## Deployment credential and Git integration

Enable Preview URLs for the `limen` Worker before staging. Its normal `workers.dev` route may remain disabled. `wrangler versions upload` reads the existing preview setting; declaring `preview_urls: true` in the configuration does not apply that setting during an inactive upload. The upload must return an immutable Version URL before staging can pass.

Use a user-scoped Cloudflare API token restricted to the Limen account, with Workers Scripts Edit and Workers Builds Configuration Read. The latter permits the activation guard to inspect build triggers; it does not modify them. Store it as `CLOUDFLARE_DOCS_API_TOKEN` in the GitHub `documentation` environment. Do not commit or print the token.

Keep the existing Git-integrated build until the replacement and its rollback are concrete. At cutover, open the `limen` Worker in Cloudflare, select **Settings → Builds → Disconnect**, and verify that its trigger inventory is empty. Disconnect only this Worker's repository integration. Do not uninstall the account's Cloudflare GitHub application.

The publisher refuses activation while any independent trigger remains. A missing credential or failed guard preserves the existing production deployment. Once the replacement passes live acceptance, remove the obsolete renderer and its build configuration; the temporary parallel configurations are not the final state.

## Evidence and recovery

The workflow retains its proof and deployment diagnostics in `released-research-book-evidence`; public Release/static artifacts provide durable anonymous access. Inspect `.docs-evidence/deployment/staged.json`, `accepted.json` or `rollback.json` alongside Wrangler's structured operation receipts. A failed operation must not produce `accepted.json`.

To inspect the active publication without credentials, use `python docs/examples/verify_publication.py --base-url https://docs.vaquum.fi/limen/ --release RELEASE_TAG` from the matching source revision. The detached manifest and rollback archive restore the exact site and its evidence into a fresh directory through `publication.restore`; activation still requires anonymous verification.
