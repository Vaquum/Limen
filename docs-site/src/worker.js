import SECURITY_HEADERS from './security-headers.json';

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
