export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/') {
      return Response.redirect(`${url.origin}/limen/`, 302);
    }

    if (url.pathname === '/limen') {
      return Response.redirect(`${url.origin}/limen/`, 302);
    }

    if (!url.pathname.startsWith('/limen/')) {
      return new Response('Not found', { status: 404 });
    }

    url.pathname = url.pathname.replace(/^\/limen/, '') || '/';

    return env.ASSETS.fetch(new Request(url, request));
  },
};
