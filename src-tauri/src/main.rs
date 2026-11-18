#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::{process::{Command, Child, Stdio}, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, io::{Read, Write, BufRead, BufReader}, time::Duration};
use tauri::{Emitter, Manager};
use serde_json::Value;

#[derive(Default)]
struct Worker {
    gate: Arc<Mutex<()>>,
    child: Arc<Mutex<Option<Child>>>,
    cancelled: Arc<AtomicBool>,
}

#[tauri::command]
fn cancel_photo(worker: tauri::State<'_, Worker>) -> Result<(), String> {
    worker.cancelled.store(true, Ordering::SeqCst);
    let mut child = worker.child.lock().map_err(|_| "Worker unavailable")?;
    if let Some(process) = child.as_mut() { process.kill().map_err(|e| e.to_string())?; }
    Ok(())
}

fn python_command(script_name: &str) -> Command {
    let root = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
    let bundled_python = root.as_ref().map(|p| p.join("python/python.exe"));
    let python = std::env::var_os("JEANMEDIA_PYTHON").unwrap_or_else(|| {
        bundled_python.filter(|p| p.is_file()).map(|p| p.into_os_string()).unwrap_or_else(|| "python".into())
    });
    let mut command = Command::new(python);
    command.env("PYTHONUTF8", "1");
    let bundled_script = root.as_ref().map(|p| p.join("worker").join(script_name)).filter(|p| p.is_file());
    let script = if script_name == "photo.py" { std::env::var_os("JEANMEDIA_WORKER") } else { None };
    command.arg(script.unwrap_or_else(|| bundled_script.unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../worker").join(script_name)).into_os_string()));
    if let Some(root) = root {
        if root.join(".worker-deps").is_dir() { command.env("PYTHONPATH", root.join(".worker-deps")); }
    }
    #[cfg(windows)] {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

#[tauri::command]
async fn local_tool(action: String, data: Value) -> Result<Value, String> {
    if !["save-project", "open-project", "diagnostics", "mask"].contains(&action.as_str()) {
        return Err("Unknown local operation".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let mut child = python_command("service.py").arg(action).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        child.stdin.take().ok_or("Worker input unavailable")?.write_all(data.to_string().as_bytes()).map_err(|e| e.to_string())?;
        let result = child.wait_with_output().map_err(|e| e.to_string())?;
        if !result.status.success() { return Err(String::from_utf8_lossy(&result.stderr).trim().to_owned()); }
        serde_json::from_slice(&result.stdout).map_err(|e| e.to_string())
    }).await.map_err(|e| e.to_string())?
}

#[tauri::command]
async fn photo(
    worker: tauri::State<'_, Worker>,
    app: tauri::AppHandle,
    source: String,
    enhance: bool,
    auto_edit: Option<bool>,
    clean_backdrop: Option<bool>,
    remove_edge_equipment: Option<[f64; 3]>,
    auto_remove_equipment: Option<bool>,
    equipment_plan: Option<Value>,
    clean_studio: Option<bool>,
    studio_plan: Option<Value>,
    clean_floor: Option<bool>,
    floor_mode: Option<String>,
    output: Option<String>,
    dpi: Option<u32>,
    quality: Option<u32>,
    approved: Option<bool>,
    inspect_region: Option<[u32; 4]>,
    mask_output: Option<String>,
) -> Result<Value, String> {
    let gate = Arc::clone(&worker.gate);
    let active = Arc::clone(&worker.child);
    let cancelled = Arc::clone(&worker.cancelled);
    tauri::async_runtime::spawn_blocking(move || {
        let _permit = gate.try_lock().map_err(|_| "Another photo is processing.")?;
        cancelled.store(false, Ordering::SeqCst);
        let mut command = python_command("photo.py");
        command.arg(source);
        let mut request = serde_json::Map::new();
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        if enhance {
            command.arg("--enhance");
        }
        if auto_edit == Some(true) {
            command.arg("--auto-edit");
        }
        if clean_backdrop == Some(true) {
            command.arg("--clean-backdrop");
        }
        if let Some(region) = remove_edge_equipment {
            command.arg("--remove-edge-equipment");
            command.args(region.map(|value| value.to_string()));
        }
        if auto_remove_equipment == Some(true) {
            command.arg("--auto-remove-equipment");
        }
        if let Some(plan) = equipment_plan {
            request.insert("equipment_plan".into(), plan);
        }
        if clean_studio == Some(true) {
            command.arg("--clean-studio");
        }
        if let Some(plan) = studio_plan {
            request.insert("studio_plan".into(), plan);
        }
        if clean_floor == Some(true) {
            command.arg("--clean-floor");
        }
        if let Some(mode) = floor_mode {
            command.arg("--floor-mode").arg(mode);
        }
        if let Some(path) = output {
            command.arg("--output").arg(path);
        }
        if let Some(dpi) = dpi {
            command.arg("--dpi").arg(dpi.to_string());
        }
        if let Some(value) = quality { command.arg("--quality").arg(value.to_string()); }
        if approved == Some(true) { command.arg("--approved"); }
        if let Some(region) = inspect_region { command.arg("--inspect-region").args(region.map(|x| x.to_string())); }
        if let Some(path) = mask_output { command.arg("--mask-output").arg(path); }
        command.arg("--request-stdin").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|e| format!("Could not start photo worker: {e}"))?;
        let mut stdin = child.stdin.take().ok_or("Worker input unavailable")?;
        let mut stdout = child.stdout.take().ok_or("Worker output unavailable")?;
        let stderr = child.stderr.take().ok_or("Worker errors unavailable")?;
        *active.lock().map_err(|_| "Worker unavailable")? = Some(child);
        let writer = std::thread::spawn(move || stdin.write_all(Value::Object(request).to_string().as_bytes()));
        let reader = std::thread::spawn(move || { let mut bytes = Vec::new(); stdout.read_to_end(&mut bytes).map(|_| bytes) });
        let errors = std::thread::spawn(move || {
            let mut messages = String::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Some(json) = line.strip_prefix("@progress:") {
                    if let Ok(value) = serde_json::from_str::<Value>(json) { let _ = app.emit("photo-progress", value); }
                } else { messages.push_str(&line); messages.push('\n'); }
            }
            messages
        });
        let status = loop {
            let mut child = active.lock().map_err(|_| "Worker unavailable")?;
            let process = child.as_mut().ok_or("Worker unavailable")?;
            if cancelled.load(Ordering::SeqCst) { let _ = process.kill(); }
            if let Some(status) = process.try_wait().map_err(|e| e.to_string())? {
                *child = None;
                break status;
            }
            drop(child);
            std::thread::sleep(Duration::from_millis(80));
        };
        let stdout = reader.join().map_err(|_| "Worker output failed")?.map_err(|e| e.to_string())?;
        let _ = writer.join();
        let stderr = errors.join().map_err(|_| "Worker errors failed")?;
        if cancelled.load(Ordering::SeqCst) { return Err("Cancelled".into()); }
        if !status.success() { return Err(stderr.trim().to_owned()); }
        serde_json::from_slice(&stdout).map_err(|e| e.to_string())
    }).await.map_err(|e| e.to_string())?
}

fn main() {
    let context = tauri::generate_context!();
    // Portable releases need their own writable WebView profile. Sharing the
    // installed profile with an older executable can prevent startup entirely.
    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", directory.join("WebViewData"));
        }
    }
    tauri::Builder::default()
        .manage(Worker::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![photo, cancel_photo, local_tool])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let worker = window.state::<Worker>();
                worker.cancelled.store(true, Ordering::SeqCst);
                if let Ok(mut child) = worker.child.lock() {
                    if let Some(process) = child.as_mut() { let _ = process.kill(); }
                };
            }
        })
        .run(context)
        .expect("could not start JeanMedia");
}

