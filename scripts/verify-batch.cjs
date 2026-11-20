const {chromium}=require(process.env.JEANMEDIA_PLAYWRIGHT);
const fs=require('node:fs');const assert=require('node:assert/strict');
const image='data:image/png;base64,'+fs.readFileSync('qa/floor20-9369-fixed/IMG_9369-after.png').toString('base64');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1180,height:850}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
  let batch=false,firstResolve,failedOnce=false;const calls=[],reports=[];
  await page.addInitScript(()=>{window.__TAURI_INTERNALS__={metadata:{currentWindow:{label:'main'},currentWebview:{label:'main'}}};});
  await page.exposeFunction('workerInvoke',async(command,args)=>{
   if(command.startsWith('plugin:dialog|open'))return batch?'D:/fixtures':['D:/fixtures/solo.jpg'];
   if(command==='local_tool'){
    if(args.action==='diagnostics')return {};
    if(args.action==='batch-list')return {folder:'D:/fixtures',sources:['D:/fixtures/one.jpg','D:/fixtures/two.jpg','D:/fixtures/three.jpg']};
    if(args.action==='batch-output')return {output:args.data.source.replace('fixtures/','fixtures/edited/')};
    if(args.action==='batch-report'){reports.push(args.data.items);return {report:'D:/fixtures/edited/report.json'};}
   }
   if(command==='photo'){
    if(!args.autoEdit){calls.push('load:'+args.source);return {width:4000,height:6000,before:image,after:image,backdropStatus:''};}
    assert.equal(args.compactOutput,true);assert.equal(args.enhance,true);assert.equal(args.approved,true);assert(args.output.includes('/edited/'));
    calls.push('edit:'+args.source);
    if(args.source.endsWith('one.jpg'))await new Promise(resolve=>{firstResolve=resolve;});
    if(args.source.endsWith('two.jpg')&&!failedOnce){failedOnce=true;throw new Error('Unreadable photo');}
    return {saved:true,review:args.source.endsWith('one.jpg')?['Floor needs review']:[]};
   }
   throw new Error('Unhandled IPC '+command);
  });
  await page.route('**/src/main.tsx*',async route=>{const response=await route.fetch();await route.fulfill({response,body:"import {mockIPC,mockWindows} from '/node_modules/@tauri-apps/api/mocks.js';mockIPC((cmd,args)=>window.workerInvoke(cmd,args),{shouldMockEvents:true});mockWindows('main');\n"+await response.text()});});
  await page.goto('http://127.0.0.1:1423');
  await page.getByRole('heading',{name:'Choose your editing mode'}).waitFor();await page.screenshot({path:'qa/batch-home.png'});
  await page.getByRole('button',{name:/^Solo Edit/}).click();await page.getByRole('button',{name:'Open photos',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.notice progress'));
  await page.getByRole('img',{name:'Original photo',exact:true}).waitFor();
  await page.getByRole('button',{name:'Modes',exact:true}).click();batch=true;
  await page.getByRole('button',{name:/^Batch Edit/}).click();await page.getByRole('button',{name:'Choose folder'}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.batch-queue li').length===3);
  assert.equal(calls.filter(x=>!x.includes('solo')).length,0,'Choosing folder must not preload all photos');
  await page.getByRole('button',{name:'Start batch',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.batch-queue li[data-status=processing]'));
  while(!firstResolve)await new Promise(resolve=>setTimeout(resolve,20));
  assert.deepEqual(calls.filter(x=>x.startsWith('edit:')),['edit:D:/fixtures/one.jpg']);
  await page.getByRole('button',{name:'Stop after current',exact:true}).click();firstResolve();
  await page.getByRole('button',{name:'Resume batch',exact:true}).waitFor();
  assert.deepEqual(calls.filter(x=>x.startsWith('edit:')),['edit:D:/fixtures/one.jpg']);
  assert.equal(await page.locator('.batch-queue li[data-status=saved]').count(),1);
  await page.getByRole('button',{name:'Modes',exact:true}).click();await page.getByRole('button',{name:/^Solo Edit/}).click();
  assert.equal(await page.locator('.queue button').count(),1,'Solo session survives switching modes');
  await page.getByRole('img',{name:'Original photo',exact:true}).waitFor();
  await page.getByRole('button',{name:'Modes',exact:true}).click();await page.getByRole('button',{name:/^Batch Edit/}).click();
  await page.getByRole('button',{name:'Resume batch',exact:true}).click();
  await page.getByRole('button',{name:'Retry failed',exact:true}).waitFor();
  assert.deepEqual(calls.filter(x=>x.startsWith('edit:')),['edit:D:/fixtures/one.jpg','edit:D:/fixtures/two.jpg','edit:D:/fixtures/three.jpg']);
  assert.equal(await page.locator('.batch-queue li[data-status=saved]').count(),2);assert.equal(await page.locator('.batch-queue li[data-status=failed]').count(),1);
  await page.getByRole('button',{name:'Retry failed',exact:true}).click();await page.getByRole('button',{name:'Finished',exact:true}).waitFor();
  assert.equal(await page.locator('.batch-queue li[data-status=saved]').count(),3);assert.equal(reports.at(-1).filter(x=>x.status==='saved').length,3);
  assert.equal(calls.filter(x=>x==='edit:D:/fixtures/one.jpg').length,1,'Resume must not re-edit saved files');
  assert.equal(errors.length,0,errors.join('\n'));await page.screenshot({path:'qa/batch-finished.png'});
  console.log(JSON.stringify({passed:true,sequential:true,automaticSave:true,stopResume:true,failedFileContinues:true,retryFailed:true,soloSessionPreserved:true,reports:reports.length}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
