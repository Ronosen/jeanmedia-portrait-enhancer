import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { PhotoViewport } from "./PhotoViewport";

type Row = {source: string; status: "pending" | "processing" | "saved" | "failed"; output?: string; review?: string[]; error?: string};
type Preview = {after: string};
type Progress = {stage: string; percent: number};
const name = (path: string) => path.split(/[\\/]/).pop() || path;

export function BatchEditor({onBack, dpi, quality, setDpi, setQuality}: {
  onBack: () => void; dpi: number; quality: number; setDpi: (value: number) => void; setQuality: (value: number) => void;
}) {
  const [folder, setFolder] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const rowsRef = useRef<Row[]>([]);
  const queueRef = useRef<HTMLOListElement>(null);
  const [current, setCurrent] = useState(-1);
  const [preview, setPreview] = useState<string>();
  const [progress, setProgress] = useState<Progress>();
  const [running, setRunning] = useState(false);
  const [picking, setPicking] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [notice, setNotice] = useState("");
  const lock = useRef(false);
  const stop = useRef(false);
  const runId = useRef("");
  const saved = rows.filter(row => row.status === "saved").length;
  const failed = rows.filter(row => row.status === "failed").length;
  const review = rows.filter(row => row.status === "saved" && row.review?.length).length;
  const remaining = rows.length - saved;
  const valid = Number.isInteger(dpi) && dpi >= 1 && dpi <= 2400 && Number.isInteger(quality) && quality >= 50 && quality <= 100;
  function update(next: Row[]) {rowsRef.current = next; setRows(next);}
  function patch(index: number, change: Partial<Row>) {update(rowsRef.current.map((row, i) => i === index ? {...row, ...change} : row));}
  useEffect(() => {
    const listener = listen<Progress>("photo-progress", event => {if (lock.current) setProgress(event.payload);});
    return () => {void listener.then(unlisten => unlisten());};
  }, []);
  useEffect(() => {queueRef.current?.children.item(current)?.scrollIntoView({block: "nearest"});}, [current]);
  async function chooseFolder() {
    if (lock.current || picking) return;
    setPicking(true); setNotice("");
    try {
      const selected = await open({directory: true, multiple: false});
      if (typeof selected !== "string") return;
      const result = await invoke<{folder: string; sources: string[]}>("local_tool", {action: "batch-list", data: {folder: selected}});
      setFolder(result.folder); update(result.sources.map(source => ({source, status: "pending"})));
      runId.current = crypto.randomUUID(); setCurrent(-1); setPreview(undefined); setProgress(undefined);
      if (!result.sources.length) setNotice("No supported photos in this folder");
    } catch (error) {setNotice(String(error));}
    finally {setPicking(false);}
  }
  async function report() {
    await invoke("local_tool", {action: "batch-report", data: {folder, runId: runId.current,
      items: rowsRef.current.map(row => ({source: name(row.source), status: row.status,
        output: row.output ? name(row.output) : undefined, review: row.review || [], error: row.error}))}});
  }
  async function start() {
    if (lock.current || !remaining || !valid) return;
    lock.current = true; stop.current = false; setRunning(true); setStopping(false); setNotice("");
    let stopReason = "";
    try {
      await report(); // Check the destination before spending time editing.
      for (let i = 0; i < rowsRef.current.length; i++) {
        if (stop.current) break;
        const row = rowsRef.current[i];
        if (row.status === "saved") continue;
        setCurrent(i); setPreview(undefined); setProgress(undefined);
        patch(i, {status: "processing", error: undefined});
        try {
          const original = await invoke<Preview>("photo", {source: row.source, enhance: false});
          setPreview(original.after);
          const {output} = await invoke<{output: string}>("local_tool", {action: "batch-output", data: {source: row.source}});
          const result = await invoke<{saved: boolean; review?: string[]; studioPlan?: {review?: string[]}}>("photo", {
            source: row.source, enhance: true, autoEdit: true, output, dpi, quality, approved: true, compactOutput: true,
          });
          if (!result.saved) throw new Error("Photo was not saved");
          patch(i, {status: "saved", output, review: result.review || result.studioPlan?.review || [], error: undefined});
        } catch (error) {
          const message = String(error); patch(i, {status: "failed", error: message});
          if (/no space|disk full|not enough space|errno 28/i.test(message)) {stop.current = true; stopReason = "Stopped: not enough disk space"; setNotice(stopReason);}
        }
        await report(); // Each saved result is recorded before opening the next.
      }
      const latest = rowsRef.current;
      const count = latest.filter(row => row.status === "saved").length;
      const errors = latest.filter(row => row.status === "failed").length;
      setNotice((stop.current ? "Stopped · " : "Finished · ") + count + " saved · " + errors + " failed");
      const last = [...latest].reverse().find(row => row.status === "saved" && row.output);
      if (last?.output) {
        try {const image = await invoke<Preview>("photo", {source: last.output, enhance: false}); setPreview(image.after); setCurrent(latest.indexOf(last));}
        catch { /* Saved files and report remain available if the display fails. */ }
      }
    } catch (error) {setNotice(String(error));}
    finally {lock.current = false; setRunning(false); setStopping(false); setProgress(undefined);}
  }
  return <main>
    <header><strong>JeanMedia <small>Batch Edit</small></strong><div className="actions">
      <button disabled={running || picking} onClick={chooseFolder}>Choose folder</button>
      <button disabled={running || picking} onClick={onBack}>Modes</button>
    </div></header>
    <section className="batch-summary" aria-label="Batch progress">
      <div><strong>{saved} / {rows.length} saved</strong><span>{failed} failed · {review} review</span></div>
      {folder && <p className="batch-path">{folder} / edited</p>}
      <progress aria-label="Overall batch progress" max={rows.length || 1} value={saved + failed} />
      <div className="actions">
        <button className="primary" disabled={running || picking || !remaining || !valid} onClick={() => void start()}>
          {!remaining && rows.length ? "Finished" : failed && saved + failed === rows.length ? "Retry failed" : saved || failed ? "Resume batch" : "Start batch"}
        </button>
        {running && <button disabled={stopping} onClick={() => {stop.current = true; setStopping(true);}}>Stop after current</button>}
        <span role="status">{stopping ? "Stopping after this photo…" : notice}</span>
      </div>
      <details className="batch-settings"><summary>Export settings</summary><div className="actions">
        <label>PPI <input disabled={running} type="number" min={1} max={2400} value={dpi} onChange={event => setDpi(Number(event.target.value))} /></label>
        <label>JPEG quality <input disabled={running} type="number" min={50} max={100} value={quality} onChange={event => setQuality(Number(event.target.value))} /></label>
      </div></details>
    </section>
    <section className="batch-workspace">
      <ol ref={queueRef} className="batch-queue" aria-label="Batch queue">{rows.map((row, i) => <li key={row.source} aria-current={i === current} data-status={row.status}>
        <span>{name(row.source)}</span><strong>{row.status === "saved" ? row.review?.length ? "Saved · Review" : "Saved" : row.status === "processing" ? "Editing…" : row.status === "failed" ? "Failed" : "Waiting"}</strong>
        {row.error && <small>{row.error}</small>}
      </li>)}</ol>
      <div className="batch-photo editor">
        <div className="toolbar">{current >= 0 ? name(rows[current]?.source || "") : "Choose a photo folder"}
          {running && <><span>{progress?.stage || "Opening photo…"}</span><progress aria-label="Current photo progress" max={100} value={progress?.percent || 0} /></>}
        </div>
        {preview ? <PhotoViewport key={String(current)}><div className="comparison"><img src={preview} alt="Batch photo" /></div></PhotoViewport> : <div className="empty">{running ? "Opening photo…" : "Batch Edit"}</div>}
      </div>
    </section>
  </main>;
}
