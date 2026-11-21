import { useEffect, useRef, useState, type CSSProperties } from "react";
import { createRoot } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { PhotoViewport } from "./PhotoViewport";
import { BatchEditor } from "./BatchEditor";
import "./style.css";

type Plan = { review?: string[]; [key: string]: unknown };
type Preview = { width: number; height: number; before: string; after: string; studioPlan?: Plan; backdropStatus: string; review?: string[] };
type Item = { source: string; original?: Preview; result?: Preview; edited: boolean; approved: boolean; error?: string };
type Progress = { stage: string; percent: number };
type Project = { source: string; studioPlan?: Plan; enhance: boolean; approved: boolean };
const formats = [{ name: "Photos", extensions: ["jpg", "jpeg", "png", "tif", "tiff"] }];
const projects = [{ name: "JeanMedia project", extensions: ["jeanmedia"] }];
const filename = (path: string) => path.split(/[\\/]/).pop() || "photo";
const needsEdit = (item: Item) => !!item.original && (!item.edited || item.result?.studioPlan?.processingRevision !== 22);
function preference(key: string, fallback: number) { try { return Number(localStorage.getItem(key)) || fallback; } catch { return fallback; } }

function App() {
  const [mode, setMode] = useState<"home" | "solo" | "batch">("home");
  const modeRef = useRef(mode); modeRef.current = mode;
  const [items, setItems] = useState<Item[]>([]);
  const currentItems = useRef<Item[]>([]);
  const [index, setIndex] = useState(0);
  const [split, setSplit] = useState(0);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [progress, setProgress] = useState<Progress>();
  const [menu, setMenu] = useState(false);
  const [diagnostic, setDiagnostic] = useState<Record<string, unknown>>();
  const [dpi, setDpi] = useState(() => preference("dpi", 300));
  const [quality, setQuality] = useState(() => preference("quality", 96));
  const [inspection, setInspection] = useState<{before: string; after: string}>();
  const [inspectionOriginal, setInspectionOriginal] = useState(false);
  const locked = useRef(false);
  const cancelled = useRef(false);
  const file = items[index];
  const preview = file?.edited ? file.result : file?.original;
  const review = file?.edited ? file.result?.studioPlan?.review || file.result?.review || [] : [];
  const exportValid = Number.isInteger(dpi) && dpi >= 1 && dpi <= 2400 && Number.isInteger(quality) && quality >= 50 && quality <= 100;
  function update(next: Item[]) { currentItems.current = next; setItems(next); }
  function patchItem(i: number, patch: Partial<Item>) { update(currentItems.current.map((item, j) => j === i ? {...item, ...patch} : item)); }
  async function run(label: string, work: () => Promise<void>) {
    if (locked.current) return;
    locked.current = true; cancelled.current = false;
    setBusy(label); setNotice(""); setProgress(undefined);
    try { await work(); }
    catch (error) { setNotice(String(error)); }
    finally { locked.current = false; setBusy(""); setProgress(undefined); }
  }
  async function load(paths: string[]) {
    const unique = paths.filter((path, i) => paths.indexOf(path) === i);
    const next = unique.map(source => ({source, edited: false, approved: false}));
    update(next); setIndex(0); setSplit(0); setInspection(undefined);
    for (let i = 0; i < next.length; i++) {
      if (cancelled.current) break;
      try { patchItem(i, { original: await invoke<Preview>("photo", { source: next[i].source, enhance: false }) }); }
      catch (error) { patchItem(i, {error: String(error)}); }
    }
  }
  useEffect(() => {
    const drops = getCurrentWebview().onDragDropEvent(({ payload }) => {
      if (payload.type === "drop" && modeRef.current === "solo") void run("Opening…", () => load(payload.paths));
    });
    const stages = listen<Progress>("photo-progress", event => setProgress(event.payload));
    void invoke<Record<string, unknown>>("local_tool", {action: "diagnostics", data: {}}).then(setDiagnostic).catch(() => {});
    return () => { void drops.then(fn => fn()); void stages.then(fn => fn()); };
  }, []);
  useEffect(() => { setInspection(undefined); setSplit(0); }, [index]);
  useEffect(() => { try { localStorage.setItem("dpi", String(dpi)); localStorage.setItem("quality", String(quality)); } catch {} }, [dpi, quality]);
  async function editAll() {
    for (let i = 0; i < currentItems.current.length; i++) {
      if (cancelled.current) break;
      const item = currentItems.current[i];
      if (!needsEdit(item)) continue;
      setIndex(i); setInspection(undefined);
      try {
        const result = await invoke<Preview>("photo", {source: item.source, enhance: true, autoEdit: true});
        if (!result.studioPlan) throw new Error("Cleaning could not finish");
        patchItem(i, {result, edited: true, approved: false, error: undefined});
      } catch (error) { patchItem(i, {error: String(error)}); }
    }
    const failed=currentItems.current.filter(item => item.error).length;
    const pending=currentItems.current.filter(item => item.edited && !!item.result?.studioPlan?.review?.length && !item.approved).length;
    setNotice(cancelled.current ? "Cancelled" : failed ? `${failed} need attention` : pending ? `${pending} need review` : "Done");
  }
  async function savePhoto(item: Item, output: string) {
    await invoke("photo", {source: item.source, enhance: item.edited,
      studioPlan: item.edited ? item.result?.studioPlan : undefined, output, dpi, quality, approved: item.approved});
  }
  return <>
    {mode === "home" && <main className="mode-page"><header><strong>JeanMedia <small>Offline</small></strong></header>
      <section className="mode-choices" aria-label="Editing mode"><h1>Choose your editing mode</h1><div>
        <button onClick={() => setMode("solo")}><strong>Solo Edit</strong><span>Edit and review photos</span></button>
        <button onClick={() => setMode("batch")}><strong>Batch Edit</strong><span>Edit a folder automatically</span></button>
      </div></section>
    </main>}
    <div hidden={mode !== "batch"}><BatchEditor onBack={() => setMode("home")} dpi={dpi} quality={quality} setDpi={setDpi} setQuality={setQuality} /></div>
    <div hidden={mode !== "solo"}><main>
    <header><strong>JeanMedia <small>Offline</small></strong><div className="actions">
      <button disabled={!!busy} onClick={() => setMode("home")}>Modes</button>
      <button disabled={!!busy} onClick={() => void run("Opening…", async () => {
        const paths = await open({filters: formats, multiple: true});
        if (paths) await load(typeof paths === "string" ? [paths] : paths);
      })}>Open photos</button>
      <button disabled={!preview || !!busy || !exportValid || (!!review.length && !file.approved)} onClick={() => void run("Saving…", async () => {
        const output = await save({filters: formats, defaultPath: filename(file.source).replace(/\.[^.]+$/, "")+"-finished.jpg"});
        if (output) { await savePhoto(file, output); setNotice("Saved"); }
      })}>Save</button>
      <button aria-label="Project and settings" aria-expanded={menu} disabled={!!busy} onClick={() => setMenu(!menu)}>⋯</button>
    </div></header>
    {(busy || notice) && <div className="notice" role="status">{progress && busy ? progress.stage : busy || notice}
      {busy && <><progress max={100} value={progress?.percent || 0} /><button onClick={() => {
        cancelled.current = true; void invoke("cancel_photo").catch(error => setNotice(String(error)));
      }}>Cancel</button></>}
    </div>}
    {menu && <section className="settings" aria-label="Project and settings">
      <div className="actions">
        <button disabled={!!busy} onClick={() => void run("Opening project…", async () => {
          const path = await open({filters: projects}); if (typeof path !== "string") return;
          const project = await invoke<Project>("local_tool", {action: "open-project", data: {path}});
          await load([project.source]);
          if (project.studioPlan) {
            const result = await invoke<Preview>("photo", {source: project.source, enhance: project.enhance, studioPlan: project.studioPlan});
            patchItem(0, {result, edited: true, approved: project.approved});
          }
          setMenu(false);
        })}>Open project</button>
        <button disabled={!preview || !!busy} onClick={() => void run("Saving project…", async () => {
          const path = await save({filters: projects, defaultPath: filename(file.source)+".jeanmedia"}); if (!path) return;
          await invoke("local_tool", {action: "save-project", data: {path, source: file.source, enhance: file.edited,
            studioPlan: file.edited ? file.result?.studioPlan : undefined, approved: file.approved,
            history: [{operation: "import"}, ...(file.edited ? [{operation: "auto-edit"}] : [])]}});
          setNotice("Project saved");
        })}>Save project</button>
        <button disabled={!items.length || !!busy} onClick={() => { update([]); setIndex(0); setInspection(undefined); setNotice(""); }}>Clear session</button>
      </div>
      <div className="actions"><label>PPI <input type="number" min={1} max={2400} value={dpi} onChange={e => setDpi(Number(e.target.value))} /></label>
        <label>JPEG quality <input type="number" min={50} max={100} value={quality} onChange={e => setQuality(Number(e.target.value))} /></label>
        <span>{preview && exportValid ? `${(preview.width/dpi*2.54).toFixed(1)} × ${(preview.height/dpi*2.54).toFixed(1)} cm` : ""}</span>
      </div>
      <div className="actions">
        <button disabled={!!busy || !exportValid || !items.some(item => item.edited)} onClick={() => void run("Saving queue…", async () => {
          const folder = await open({directory: true}); if (typeof folder !== "string") return;
          let saved = 0, skipped = 0;
          for (let i = 0; i < currentItems.current.length; i++) {
            if (cancelled.current) break;
            const item = currentItems.current[i];
            if (!item.edited || ((item.result?.studioPlan?.review?.length || 0) > 0 && !item.approved)) { skipped++; continue; }
            try { await savePhoto(item, folder+"/"+String(i+1).padStart(3,"0")+"-"+filename(item.source).replace(/\.[^.]+$/, "")+"-finished.jpg"); saved++; }
            catch (error) { patchItem(i,{error:String(error)}); skipped++; }
          }
          setNotice(`${saved} saved · ${skipped} skipped`);
        })}>Save queue</button>
      </div>
      <details><summary>Hardware</summary><pre>{diagnostic ? JSON.stringify(diagnostic,null,2) : "Checking…"}</pre></details>
    </section>}
    <div className="workspace simple" style={{"--review-space": review.length ? (file.approved ? "72px" : "170px") : "0px"} as CSSProperties}>
      {!!items.length && <nav className="queue" aria-label="Photo queue">{items.map((item,i) => <button key={item.source} disabled={!!busy} aria-current={i===index} onClick={() => setIndex(i)}>
        {filename(item.source)} {item.error ? "!" : item.edited && !!item.result?.studioPlan?.review?.length && !item.approved ? "Review" : item.edited ? "✓" : ""}
      </button>)}</nav>}
      {file?.error && <p className="notice" role="alert">{file.error}</p>}
      {!!review.length && <div className="review">{!file.approved && <><span>Needs review</span><ul>{review.map((text,i) => <li key={i}>{text}</li>)}</ul></>}
        <label><input type="checkbox" disabled={!!busy} checked={file.approved} onChange={e => patchItem(index,{approved:e.target.checked})} />{file.approved ? "Approved for saving" : "Approve this result for saving"}</label>
      </div>}
      <section className="editor" aria-label="Photo workspace">
        {preview ? <>
          <PhotoViewport key={file.source}><div className="comparison" onDoubleClick={e => {
            const box=e.currentTarget.getBoundingClientRect();
            const w=Math.min(1000,preview.width),h=Math.min(800,preview.height);
            const x=Math.max(0,Math.min(preview.width-w,Math.round((e.clientX-box.left)/box.width*preview.width-w/2)));
            const y=Math.max(0,Math.min(preview.height-h,Math.round((e.clientY-box.top)/box.height*preview.height-h/2)));
            void run("Inspecting…",async () => setInspection(await invoke("photo",{source:file.source,enhance:file.edited,
              studioPlan:file.edited?file.result?.studioPlan:undefined,inspectRegion:[x,y,w,h]})));
          }}>
            <img src={preview.after} alt={file.edited ? "Edited photo" : "Original photo"} />
            {file.edited && <><img className="before" src={preview.before} alt="Original photo" style={{clipPath:`inset(0 ${100-split}% 0 0)`}} />{split>0 && split<100 && <div className="divider" style={{left:`${split}%`}} />}</>}
          </div></PhotoViewport>
          <div className="comparison-controls">{file.edited && <><span>Before</span><input aria-label="Before and after" type="range" min={0} max={100} value={split} onChange={e => setSplit(Number(e.target.value))} /><span>After</span></>}
            <button disabled={!!busy} onClick={() => void run("Inspecting…", async () => {
              const w=Math.min(1000,preview.width),h=Math.min(800,preview.height);
              setInspection(await invoke("photo",{source:file.source,enhance:file.edited,
                studioPlan:file.edited?file.result?.studioPlan:undefined,
                inspectRegion:[Math.floor((preview.width-w)/2),Math.floor((preview.height-h)/2),w,h]}));
            })}>100%</button>
          </div>
        </> : <div className="empty"><h1>Drop photos here</h1></div>}
      </section>
      <div className="finish-controls">
        <button className="primary" disabled={!items.some(needsEdit) || !!busy} onClick={() => void run("Editing…",editAll)}>Auto Edit</button>
        <button disabled={!file?.edited || !!busy} onClick={() => {patchItem(index,{edited:false,approved:false});}}>Undo</button>
        <button disabled={file?.edited || !file?.result || !!busy} onClick={() => {patchItem(index,{edited:true});setSplit(0);}}>Redo</button>
      </div>
    </div>
    {inspection && <div className="inspection" role="dialog" aria-label="100 percent inspection"><div className="actions"><strong>100%</strong><button onClick={() => setInspectionOriginal(!inspectionOriginal)}>{inspectionOriginal?"After":"Before"}</button><button onClick={() => setInspection(undefined)}>Close</button></div><div className="pixel-view"><img src={inspectionOriginal?inspection.before:inspection.after} alt="Full resolution inspection" /></div></div>}
  </main></div></>;
}
createRoot(document.getElementById("root")!).render(<App />);
