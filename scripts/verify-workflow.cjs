const {chromium}=require(process.env.JEANMEDIA_PLAYWRIGHT);
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const root=process.cwd();
const python=process.env.JEANMEDIA_TEST_PYTHON;
let dialogOpen=['D:/test22/IMG_0881.JPG','D:/test22/IMG_0779.JPG'];
if(process.env.JEANMEDIA_UI_PLANS_DIR)dialogOpen=['D:/test22/IMG_0164.JPG','D:/test22/IMG_0203.JPG'];
let dialogSave=path.join(root,'qa','workflow.jeanmedia');
for(const name of ['workflow.jeanmedia','workflow-export.png']){const target=path.join(root,'qa',name);if(fs.existsSync(target))fs.unlinkSync(target);}
let active;
let lastResult;
const env={...process.env,PYTHONPATH:path.join(root,'.worker-deps')};
function worker(script,args,input){
 return new Promise((resolve,reject)=>{
  const child=spawn(python,[path.join(root,'worker',script),...args],{env,windowsHide:true});
  if(script==='photo.py')active=child;
  let stdout='',stderr='';
  child.stdout.on('data',chunk=>stdout+=chunk);
  child.stderr.on('data',chunk=>stderr+=chunk);
  child.on('error',reject);
  child.on('close',code=>{
   if(active===child)active=null;
   if(code!==0)return reject(new Error(stderr.split('\n').filter(x=>!x.startsWith('@progress:')).join('\n')||'Cancelled'));
   try {const result=JSON.parse(stdout);if(script==='photo.py')lastResult=result;resolve(result);}catch(error){reject(error);}
  });
  child.stdin.end(input?JSON.stringify(input):'');
 });
}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try {
  const page=await browser.newPage({viewport:{width:1180,height:800}});
  await page.addInitScript(() => { window.__TAURI_INTERNALS__ = {metadata:{currentWindow:{label:'main'},currentWebview:{label:'main'}}}; });
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.exposeFunction('workerInvoke',async(cmd,args)=>{
   if(cmd.startsWith('plugin:dialog|open'))return dialogOpen;
   if(cmd.startsWith('plugin:dialog|save'))return dialogSave;
   if(cmd==='cancel_photo'){if(active)active.kill();return null;}
   if(cmd==='local_tool')return worker('service.py',[args.action],args.data);
   if(cmd==='photo'){
    if(args.autoEdit&&process.env.JEANMEDIA_UI_PLANS_DIR){
     const stem=path.basename(args.source,path.extname(args.source));
     const fixture=path.join(process.env.JEANMEDIA_UI_PLANS_DIR,stem+'-plan.json');
     if(fs.existsSync(fixture)){
      const plan=JSON.parse(fs.readFileSync(fixture,'utf8'));plan.processingRevision=15;
      args={...args,autoEdit:false,enhance:true,studioPlan:plan};
     }
    }
    const flags=[args.source,'--request-stdin'];
    const booleans={enhance:'--enhance',autoEdit:'--auto-edit',approved:'--approved'};
    for(const [key,flag] of Object.entries(booleans))if(args[key])flags.push(flag);
    const values={output:'--output',dpi:'--dpi',quality:'--quality',maskOutput:'--mask-output'};
    for(const [key,flag]of Object.entries(values))if(args[key]!==undefined)flags.push(flag,typeof args[key]==='object'?JSON.stringify(args[key]):String(args[key]));
    if(args.inspectRegion)flags.push('--inspect-region',...args.inspectRegion.map(String));
    return worker('photo.py',flags,{studio_plan:args.studioPlan||null});
   }
   throw new Error('Unhandled IPC '+cmd);
  });
  await page.route('**/src/main.tsx*',async route=>{
   const response=await route.fetch();
   await route.fulfill({response,body:`import {mockIPC,mockWindows} from '/node_modules/@tauri-apps/api/mocks.js';mockIPC((cmd,args)=>window.workerInvoke(cmd,args),{shouldMockEvents:true});mockWindows('main');\n`+await response.text()});
  });
  await page.goto('http://127.0.0.1:1420');
  console.log('UI loaded');
  await page.getByRole('button',{name:'Open photos',exact:true}).click();
  await page.getByRole('button',{name:'Auto Edit',exact:true}).waitFor();
  await page.waitForFunction(()=>!document.querySelector('progress'),null,{timeout:120000});
  if(await page.getByRole('button',{name:'Auto Edit',exact:true}).isDisabled())console.log('DISABLED STATE',await page.locator('body').innerText(),errors);
  await page.getByRole('button',{name:'Auto Edit',exact:true}).click();
  console.log('Editing queue');
  await page.waitForFunction(()=>!document.querySelector('progress'),null,{timeout:420000});
  assert.equal(await page.locator('.queue button').count(),2);
  console.log('Queue finished',await page.locator('body').innerText());
  for(let i=0;i<2;i++){await page.locator('.queue button').nth(i).click(); const approval=page.getByLabel('Approve this result for saving');if(await approval.count()){assert(await page.getByRole('button',{name:'Save',exact:true}).isDisabled());await approval.check();}}
  if(await page.locator('.queue button').filter({hasText:'✓'}).count()!==2){await page.screenshot({path:'qa/v3-workflow-failure.png'});console.log(await page.locator('body').innerText(),errors);}
  assert.equal(await page.locator('.queue button').filter({hasText:'✓'}).count(),2);
  assert.equal(await page.getByRole('button',{name:'Masks',exact:true}).count(),0);
  assert.equal(await page.getByRole('button',{name:'Save change mask',exact:true}).count(),0);
  assert.equal(await page.getByRole('button',{name:'Changes',exact:true}).count(),0);
  await page.getByRole('button',{name:'Undo',exact:true}).click();
  assert.equal(await page.locator('.change-overlay').count(),0);
  await page.getByRole('button',{name:'Redo',exact:true}).click();
  {const approval=page.getByLabel('Approve this result for saving');if(await approval.count())await approval.check();}
  await page.getByRole('button',{name:'Project and settings'}).click();
  await page.getByRole('button',{name:'Save project',exact:true}).click();
  await page.getByText('Project saved',{exact:true}).waitFor({timeout:120000});
  assert(fs.existsSync(dialogSave));
  await page.getByRole('button',{name:'Clear session',exact:true}).click();
  assert.equal(await page.locator('.queue button').count(),0);
  dialogOpen=dialogSave;
  await page.getByRole('button',{name:'Open project',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('progress'),null,{timeout:120000});
  assert.equal(await page.locator('.queue button').filter({hasText:'✓'}).count(),1);
  dialogSave=path.join(root,'qa','workflow-export.png');
  await page.getByRole('button',{name:'Save',exact:true}).click();
  await page.getByText('Saved',{exact:true}).waitFor({timeout:120000});
  assert(fs.existsSync(dialogSave));
  await page.locator('.comparison').dblclick({position:{x:80,y:100}});
  await page.getByRole('dialog',{name:'100 percent inspection'}).waitFor({timeout:120000});
  await page.getByRole('button',{name:'Close',exact:true}).click();
  const button=await page.getByRole('button',{name:'Auto Edit',exact:true}).boundingBox();
  assert(button.y+button.height <= 800, 'Auto Edit must remain visible');
  await page.screenshot({path:path.join(root,'qa','workflow-ui.png')});
  // A failed/aborted worker must not leave editing locked.
  dialogOpen=['D:/test22/IMG_1083.JPG'];
  await page.getByRole('button',{name:'Open photos',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('progress'),null,{timeout:120000});
  await page.getByRole('button',{name:'Auto Edit',exact:true}).click();
  await page.getByRole('button',{name:'Cancel',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('progress'),null,{timeout:120000});
  assert(await page.getByRole('button',{name:'Auto Edit',exact:true}).isEnabled());
  assert.deepEqual(errors,[]);
  console.log('PASS: two-photo queue, internal masks only, undo/redo, project roundtrip, full-res export/inspection, cancellation and retry availability');
 }finally{if(active)active.kill();await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
