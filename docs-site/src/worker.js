const SECURITY_HEADERS = {
  'Content-Security-Policy': "default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline'; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=()',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
};

function withSecurityHeaders(response) {
  const secured = new Response(response.body, response);
  for (const [key, value] of Object.entries(SECURITY_HEADERS)) {
    secured.headers.set(key, value);
  }
  return secured;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/') {
      return withSecurityHeaders(Response.redirect(`${url.origin}/limen/`, 302));
    }

    if (url.pathname === '/limen') {
      return withSecurityHeaders(Response.redirect(`${url.origin}/limen/`, 302));
    }

    if (url.pathname === '/robots.txt') {
      return withSecurityHeaders(Response.redirect(`${url.origin}/limen/robots.txt`, 302));
    }

    if (url.pathname === '/sitemap.xml') {
      return withSecurityHeaders(Response.redirect(`${url.origin}/limen/sitemap.xml`, 302));
    }

    if (!url.pathname.startsWith('/limen/')) {
      return withSecurityHeaders(new Response('Not found', { status: 404 }));
    }

    url.pathname = url.pathname.replace(/^\/limen/, '') || '/';

    return withSecurityHeaders(await env.ASSETS.fetch(new Request(url, request)));
  },
};
