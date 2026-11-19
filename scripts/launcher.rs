//! Small Windows launcher for the selected, self-contained portable build.
#![windows_subsystem = "windows"]
use std::{fs, path::PathBuf, process::Command};

fn launch() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::current_exe()?.parent().ok_or("Missing app folder")?.to_path_buf();
    let relative = fs::read_to_string(root.join("APP-PATH.txt"))?;
    let relative = PathBuf::from(relative.trim().trim_start_matches('\u{feff}'));
    if relative.is_absolute() { return Err("App path must be relative".into()); }
    let target = root.join(relative).canonicalize()?;
    let releases = root.join("release").canonicalize()?;
    if !target.starts_with(releases) || target.file_name().and_then(|x| x.to_str()) != Some("jeanmedia.exe") {
        return Err("App path is outside the portable release folder".into());
    }
    let directory = target.parent().ok_or("Missing portable folder")?;
    Command::new(&target).current_dir(directory)
        .env_remove("JEANMEDIA_WORKER").env_remove("JEANMEDIA_PYTHON")
        .env_remove("PYTHONPATH").spawn()?;
    Ok(())
}

fn main() {
    if let Err(error) = launch() {
        #[link(name = "user32")]
        unsafe extern "system" { fn MessageBoxW(window: isize, text: *const u16, caption: *const u16, kind: u32) -> i32; }
        let message: Vec<u16> = format!("JeanMedia could not start: {error}\nKeep the release folder beside this launcher.")
            .encode_utf16().chain(Some(0)).collect();
        let title: Vec<u16> = "JeanMedia".encode_utf16().chain(Some(0)).collect();
        unsafe { MessageBoxW(0, message.as_ptr(), title.as_ptr(), 0x10); }
    }
}
