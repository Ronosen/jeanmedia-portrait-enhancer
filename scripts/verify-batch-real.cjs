const {chromium}=require(process.env.JEANMEDIA_PLAYWRIGHT);
const {spawn}=require('node:child_process');const fs=require('node:fs');const path=require('node:path');const crypto=require('node:crypto');const assert=require('node:assert/strict');
const root=process.cwd(),bundle=path.resolve(process.env.JEANMEDIA_BATCH_BUNDLE),folder=path.join(root,'qa','batch-native-input');
const fixtures=JSON.parse(fs.readFileSync('qa/batch-native-fixtures.json','utf8'));const calls=[],savedPhotos=[];let active=false;
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
function worker(script,args,data){return new Promise((resolve,reject)=>{
 const child=spawn(path.join(bundle,'python/python.exe'),[path.join(bundle,'worker',script),...args],{env:{...process.env,PYTHONPATH:path.join(bundle,'.worker-deps')},windowsHide:true});
 let output='',errors='';child.stdout.on('data',x=>output+=x);child.stderr.on('data',x=>errors+=x);child.on('error',reject);
 child.on('close',code=>{if(code!==0)return reject(new Error(errors.split('\n').filter(x=>!x.startsWith('@progress:')).join('\n')));try{resolve(JSON.parse(output));}catch(e){reject(e);}});child.stdin.end(JSON.stringify(data||{}));
});}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1180,height:850}});page.setDefaultTimeout(1200000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{window.__TAURI_INTERNALS__={metadata:{currentWindow:{label:'main'},currentWebview:{label:'main'}}};});
  await page.exposeFunction('workerInvoke',async(cmd,args)=>{
   if(cmd.startsWith('plugin:dialog|open'))return folder;
   if(cmd==='local_tool'){if(args.action==='diagnostics')return {};return worker('service.py',[args.action],args.data);}
   if(cmd==='photo'){
    assert(!active,'Photos must never overlap');active=true;calls.push({source:args.source,edit:!!args.autoEdit});
    const flags=[args.source,'--request-stdin'];if(args.enhance)flags.push('--enhance');if(args.autoEdit)flags.push('--auto-edit');if(args.approved)flags.push('--approved');if(args.compactOutput)flags.push('--compact-output');
    for(const key of ['output','dpi','quality'])if(args[key]!==undefined)flags.push('--'+key,String(args[key]));
    try{
     const result=await worker('photo.py',flags,{});
     if(args.output){assert.equal(result.saved,true);assert(fs.existsSync(args.output));assert(!result.studioPlan,'Batch result must be compact');savedPhotos.push({source:args.source,output:args.output,review:result.review||[]});console.log('SAVED',path.basename(args.output),result.review?.length||0);}
     return result;
    }finally{active=false;}
   }
   throw new Error('Unhandled IPC '+cmd);
  });
  await page.route('**/src/main.tsx*',async route=>{const response=await route.fetch();await route.fulfill({response,body:"import {mockIPC,mockWindows} from '/node_modules/@tauri-apps/api/mocks.js';mockIPC((cmd,args)=>window.workerInvoke(cmd,args),{shouldMockEvents:true});mockWindows('main');\n"+await response.text()});});
  await page.goto('http://127.0.0.1:1423');await page.getByRole('button',{name:/^Batch Edit/}).click();await page.getByRole('button',{name:'Choose folder'}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.batch-queue li').length===8);
  assert.equal(calls.length,0,'Folder selection must only list metadata');
  await page.getByRole('button',{name:'Start batch',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.batch-queue li[data-status=saved]').length===7&&document.querySelectorAll('.batch-queue li[data-status=failed]').length===1,null,{timeout:1800000});
  await page.getByRole('button',{name:'Retry failed',exact:true}).waitFor();await page.waitForFunction(()=>!document.querySelector('.batch-summary .primary').disabled,null,{timeout:120000});
  assert.equal(savedPhotos.length,7);assert.equal(errors.length,0,errors.join('\n'));
  for(const fixture of fixtures)assert.equal(hash(path.join(folder,fixture.name)),fixture.sha256,'Source must remain unchanged');
  assert.equal(fs.readFileSync(path.join(folder,'edited','00-IMG_9369.jpg'),'utf8'),'preserve-existing-output');
  assert(savedPhotos[0].output.endsWith('00-IMG_9369-2.jpg'));
  const reports=fs.readdirSync(path.join(folder,'edited')).filter(x=>x.startsWith('batch-')&&x.endsWith('.json'));assert.equal(reports.length,1);
  const report=JSON.parse(fs.readFileSync(path.join(folder,'edited',reports[0]),'utf8'));assert.equal(report.items.filter(x=>x.status==='saved').length,7);assert.equal(report.items.filter(x=>x.status==='failed').length,1);
  await page.screenshot({path:'qa/batch-real-finished.png'});
  fs.writeFileSync('qa/batch-real-results.json',JSON.stringify({passed:true,saved:7,intentionalErrors:1,originalsUnchanged:true,existingOutputPreserved:true,sequential:true,savedPhotos},null,2));
  console.log('PASS real seven-photo batch, automatic saves, review flags, corrupt input continuation, collision handling, original preservation');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
