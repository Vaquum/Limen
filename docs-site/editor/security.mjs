import {EditError} from './passages.mjs';
const bytes=value=>new TextEncoder().encode(value);
export const random=()=>crypto.randomUUID();
const b64=value=>Buffer.from(value).toString('base64url');
export async function challenge(value){return b64(await crypto.subtle.digest('SHA-256',bytes(value)));}
async function key(secret){if(!secret||Buffer.from(secret,'base64').length!==32)throw new EditError('Editor session key is not configured',503);return crypto.subtle.importKey('raw',Buffer.from(secret,'base64'),'AES-GCM',false,['encrypt','decrypt']);}
export async function seal(value,secret){const iv=crypto.getRandomValues(new Uint8Array(12)),encrypted=await crypto.subtle.encrypt({name:'AES-GCM',iv},await key(secret),bytes(JSON.stringify(value)));return b64(iv)+'.'+b64(encrypted);}
export async function unseal(value,secret){
  try{const [iv,cipher]=value.split('.');return JSON.parse(new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:Buffer.from(iv,'base64url')},await key(secret),Buffer.from(cipher,'base64url'))));}
  catch(error){throw new EditError('Session expired or its key was rotated; sign in again',401,{cause:error});}
}
export const cookie=(name,value,age)=>`__Host-limen-${name}=${value}; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=${age}`;
export function readCookie(request,name){return request.headers.get('Cookie')?.split(';').map(item=>item.trim()).find(item=>item.startsWith('__Host-limen-'+name+'='))?.split('=').slice(1).join('=');}
export const headers={
  'Cache-Control':'no-store','X-Robots-Tag':'noindex, nofollow','Referrer-Policy':'same-origin','X-Content-Type-Options':'nosniff',
  'Content-Security-Policy':"default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self'; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'; object-src 'none'"
};
export function response(value,status=200,extra={}){return new Response(typeof value==='string'?value:JSON.stringify(value),{status,headers:{...headers,'Content-Type':typeof value==='string'?'text/html; charset=utf-8':'application/json; charset=utf-8',...extra}});}
export function exactOrigin(request,origin){if(new URL(request.url).origin!==origin)throw new EditError('Unknown authoring origin',403);}
export async function jsonBody(request,origin){
  exactOrigin(request,origin);if(request.headers.get('Origin')!==origin)throw new EditError('Exact authoring Origin is required',403);
  if(request.headers.get('Content-Type')?.split(';')[0]!=='application/json')throw new EditError('JSON is required',415);
  const declared=Number(request.headers.get('Content-Length')||0);if(declared>2097152)throw new EditError('Request exceeds 2 MiB',413);
  const reader=request.body?.getReader();if(!reader)throw new EditError('Request body is missing');let size=0;const parts=[];
  for(;;){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>2097152){await reader.cancel();throw new EditError('Request exceeds 2 MiB',413);}parts.push(value);}
  try{return JSON.parse(Buffer.concat(parts).toString('utf8'));}catch(error){throw new EditError('Invalid JSON: '+error.message,400);}
}
