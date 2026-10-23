import {test,expect} from '@playwright/test';
const origin=process.env.EDITOR_TEST_ORIGIN||'https://127.0.0.1:4335';
const control=(request,body)=>request.post(origin+'/__test/control',{data:body});
async function signIn(page){
  await page.route('https://github.com/login/oauth/authorize**',route=>{const url=new URL(route.request().url()),callback=new URL('/limen/edit/auth/callback',origin);callback.searchParams.set('state',url.searchParams.get('state'));callback.searchParams.set('code','browser-protocol-code');return route.fulfill({status:302,headers:{Location:callback.href},body:''});});
  await page.goto(origin+'/limen/edit/?route=/developer/');await page.getByRole('link',{name:'Sign in with GitHub'}).click();await expect(page.locator('[data-frame]')).toBeVisible();await expect(page.frameLocator('[data-frame]').locator('[data-edit-id]').first()).toBeVisible();
}
async function change(page,selector,words){const target=page.frameLocator('[data-frame]').locator(selector).first();await target.click();await target.press('End');await target.pressSequentially(words);await target.press('Enter');await expect(page.locator('[data-count]')).not.toHaveText('0');return target;}
test.beforeEach(async({request})=>{await control(request,{action:'reset'});});
test('two pages, two commits and one PR preserve the cumulative writer journey',async({page,request})=>{
  const errors=[];page.on('pageerror',error=>errors.push(error.message));await signIn(page);await page.locator('[data-edit]').check();await change(page,'p[data-edit-id]',' Reviewed.');await page.getByRole('button',{name:'Commit',exact:true}).click();await expect(page.locator('[data-status]')).toContainText('Committed');
  await page.frameLocator('[data-frame]').getByRole('link',{name:'Documentation System Contract',exact:true}).first().click();await expect(page.frameLocator('[data-frame]').locator('h1')).toHaveText('Documentation system contract');await change(page,'p[data-edit-id]',' Maintained.');await page.getByRole('button',{name:'Commit',exact:true}).click();await expect(page.locator('[data-count]')).toHaveText('0');await page.getByRole('button',{name:'Open PR',exact:true}).click();await expect(page.getByRole('link',{name:'Open PR #200'})).toBeVisible();
  const data=await (await control(request,{})).json();expect(data.counts).toEqual({ref:1,commit:2,issue:1,pr:1});expect(data.changed.sort()).toEqual(['docs/Developer/Documentation-System.md','docs/Developer/README.md']);expect(errors).toEqual([]);
});
test('storage updates cannot replace the words being actively revised',async({page})=>{
  await signIn(page);await page.locator('[data-edit]').check();const target=await change(page,'p[data-edit-id]',' First revision.');await target.click();await target.press('End');await target.pressSequentially(' Still typing');const before=await target.textContent();
  await page.evaluate(()=>dispatchEvent(new StorageEvent('storage',{key:'reader:limen-authoring:any:place',newValue:'{}'})));await expect(target).toHaveText(before);await target.press('Escape');await expect(target).not.toContainText('Still typing');
});
test('lost commit replies reconcile on reload without retaining committed drafts',async({page,request})=>{
  await signIn(page);await page.locator('[data-edit]').check();await change(page,'p[data-edit-id]',' Retained revision.');
  await page.route('**/limen/edit/api/commit',async route=>{const response=await route.fetch();expect(response.status()).toBe(200);await route.abort();});await page.getByRole('button',{name:'Commit',exact:true}).click();await expect(page.locator('[data-status]')).toContainText('retained');await page.unroute('**/limen/edit/api/commit');await page.reload();await expect(page.locator('[data-count]')).toHaveText('0');await page.getByRole('button',{name:'Open PR',exact:true}).click();await expect(page.getByRole('link',{name:'Open PR #200'})).toBeVisible();const data=await (await control(request,{})).json();expect(data.counts.commit).toBe(1);
});
test('IME, keyboard cancellation, phone Undo and permission revocation retain the access boundary',async({page,request})=>{
  await signIn(page);await page.locator('[data-edit]').check();const target=page.frameLocator('[data-frame]').locator('p[data-edit-id]').first();await target.focus();await target.press('Enter');await target.press('End');await target.pressSequentially(' Unsaved');await target.dispatchEvent('keydown',{key:'Enter',isComposing:true});await expect(target).toHaveAttribute('contenteditable','true');await page.getByRole('button',{name:'Undo active edit'}).click();await expect(target).not.toContainText('Unsaved');
  await control(request,{action:'revoke'});const denied=await request.get(origin+'/limen/edit/api/session');expect(denied.status()).toBe(403);const preview=await request.get(origin+'/limen/edit/view/developer/');expect(preview.status()).toBe(403);
});
