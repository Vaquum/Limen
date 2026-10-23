# Cloudflare deployment

The Limen-owned research reader serves [docs.vaquum.fi/limen/](https://docs.vaquum.fi/limen/). Its public `publication.json` identifies the active release, source, wheel and sealed artifacts. Candidate execution alone does not establish deployment.

## Serving boundary

The `limen` Worker serves `/limen/*` through `ASSETS`. It redirects `/` and `/limen` to `/limen/`, and root robots/sitemap requests to their product resources. Other product prefixes return 404. Redirects, errors and assets retain HSTS, CSP, permissions, referrer, content-type and frame protections.

The Worker imports `src/security-headers.json`. The build retains that exact policy in its hashed site payload. Verification compares every response's six security-header values with that release's policy, including the 404 response. Rollback retains the previous release's policy.

The sole [publisher/wrangler.json](publisher/wrangler.json) reads the sealed `reader-build/`. The publisher pins Wrangler and its complete dependency graph in `publisher/package-lock.json`; it requires Node.js 22 or later. Reader checks continue to run under Node.js 20 in the required Docs Site job.

## Release-bound publication

[docs_publish.yml](../.github/workflows/docs_publish.yml) follows a successful canonical package-publishing run. Its run-scoped identity artifact names the exact merged tag, commit and wheel digest. Missing identity, failed package publication, changed source or a substituted wheel prevents deployment.

The publisher downloads the actual public PyPI wheel and the immutable project/input bundle from the matching GitHub Release. It runs the unchanged projects in a fresh environment outside the checkout, verifies complete semantic and documentation evidence, builds the reader, and runs its browser matrix. Contributor and fork checks need no deployment secrets.

After proof, the publisher seals the build and retains its source, results, wheel, environment, validation and rollback artifacts on the public GitHub Release. Uploads never replace existing bytes. The site's `publication.json` selects a detached `publication-SHA256.json`; every earlier manifest remains available for verification and rollback. Wheel downloads retain their installable distribution filename.

`deploy_publication.py stage` uploads an inactive Worker version and verifies its anonymous Version URL against those public artifacts. `activate` repeats verification, confirms that independent Git builds are disconnected and that no newer package release superseded the build, then checks the latest package release immediately before activating exactly that version. It repeats that check after deployment and after production verification. Supersession or failure restores and verifies the recorded predecessor. The first migration retains the existing legacy Worker version and home-page digest explicitly; later rollbacks also verify the predecessor's complete publication envelope.

## Deployment credential and Git integration

Enable Preview URLs for the `limen` Worker before staging. Its normal `workers.dev` route may remain disabled. `wrangler versions upload` reads the existing preview setting; declaring `preview_urls: true` in the configuration does not apply that setting during an inactive upload. The upload must return an immutable Version URL before staging can pass.

Use a user-scoped Cloudflare API token restricted to the Limen account, with Workers Scripts Edit and Workers Builds Configuration Read. The latter permits the activation guard to inspect build triggers; it does not modify them. Store it as `CLOUDFLARE_DOCS_API_TOKEN` in the GitHub `documentation` environment. Do not commit or print the token.

The Limen Worker uses release-bound publication. Its independent Git build trigger must remain disconnected. The activation guard verifies the trigger inventory; other Workers and the account’s Cloudflare GitHub application are outside this product boundary.

The publisher refuses activation while any independent trigger remains. A missing credential or failed guard preserves the existing production deployment. Every publication uses the actual PyPI wheel, matching projects and verified predecessor for rollback.

## Evidence and recovery

The workflow retains its proof and deployment diagnostics in `released-research-book-evidence`; public Release/static artifacts provide durable anonymous access. Inspect `.docs-evidence/deployment/staged.json`, `accepted.json` or `rollback.json` alongside Wrangler's structured operation receipts. A failed operation must not produce `accepted.json`.

An anonymous HTTP failure also retains `activation-http-failure.json` or `rollback-http-failure.json`. Each receipt names the phase, the requested Worker version and the release being verified. A rollback names the predecessor's release; a legacy renderer has `release: null`. Activation preflight failures receive an activation receipt without deploying or rolling back a version.

The request record contains the requested and final response URLs, HTTP status, selected response headers and a body fingerprint. URL userinfo, query strings and fragments are removed. The header set is limited to `Content-Type`, `Content-Length`, `Cache-Control`, `CF-Cache-Status`, `CF-Ray` and `ETag`. Raw bodies, server reasons, cookies and other headers are excluded from receipts and rendered HTTP exceptions.

`body_state` distinguishes `complete`, `partial` and `unavailable`. `body_bytes` and `body_sha256` describe only the observed bytes; both are null when unavailable. `body_read_error_type` identifies a failed read without its message. Partial or unavailable error bodies cannot satisfy the sealed 404 check. Complete error bodies remain available to that check with their original headers.

The sanitized observation is printed before receipt storage. Storage failure remains visible and fails the command; it cannot prevent required predecessor restoration. Each phase keeps its own observation when activation and rollback both fail, including HTTP errors wrapped by validation errors. `rollback.json` still requires successful predecessor verification. These diagnostics record the current attempt; they cannot reconstruct a request omitted from an earlier run.

To inspect the active publication without credentials, use `python docs/examples/verify_publication.py --base-url https://docs.vaquum.fi/limen/ --release RELEASE_TAG` from the matching source revision. The detached manifest and rollback archive restore the exact site and its evidence into a fresh directory through `publication.restore`; activation still requires anonymous verification.
