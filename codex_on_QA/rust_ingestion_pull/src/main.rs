use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct Config {
    source: PathBuf,
    dest: PathBuf,
    manifest: PathBuf,
}

fn print_usage() {
    eprintln!("Usage: rust_ingestion_pull --source <dir> --dest <dir> --manifest <file>");
    eprintln!("Defaults: --source=ingestion candidates, --dest=codex_on_QA/candidates, --manifest=codex_on_QA/out/latest_candidate.json");
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut source = PathBuf::from("ingestion candidates");
    let mut dest = PathBuf::from("codex_on_QA/candidates");
    let mut manifest = PathBuf::from("codex_on_QA/out/latest_candidate.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                if i + 1 < args.len() { source = PathBuf::from(&args[i+1]); i += 1; } else { print_usage(); std::process::exit(2); }
            }
            "--dest" => {
                if i + 1 < args.len() { dest = PathBuf::from(&args[i+1]); i += 1; } else { print_usage(); std::process::exit(2); }
            }
            "--manifest" => {
                if i + 1 < args.len() { manifest = PathBuf::from(&args[i+1]); i += 1; } else { print_usage(); std::process::exit(2); }
            }
            "-h" | "--help" => { print_usage(); std::process::exit(0); }
            other => {
                eprintln!("Unknown arg: {}", other);
                print_usage();
                std::process::exit(2);
            }
        }
        i += 1;
    }

    Config { source, dest, manifest }
}

fn is_candidate_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
        Some(ext) if ["odt", "pdf", "doc", "docx"].contains(&ext.as_str()) => true,
        _ => false,
    }
}

fn latest_file(dir: &Path) -> io::Result<Option<PathBuf>> {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    for entry_res in fs::read_dir(dir)? {
        let entry = entry_res?;
        let path = entry.path();
        if !path.is_file() { continue; }
        if !is_candidate_file(&path) { continue; }
        let meta = entry.metadata()?;
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        match &best {
            Some((best_time, _)) => {
                if mtime > *best_time { best = Some((mtime, path)); }
            }
            None => best = Some((mtime, path)),
        }
    }
    Ok(best.map(|(_, p)| p))
}

fn ensure_dir(path: &Path) -> io::Result<()> {
    if let Some(p) = path.parent() { fs::create_dir_all(p)?; }
    Ok(())
}

fn write_manifest(manifest_path: &Path, source: &Path, dest: &Path, mtime: SystemTime) -> io::Result<()> {
    ensure_dir(manifest_path)?;
    let ts = mtime.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let mut file = File::create(manifest_path)?;
    // Manual JSON assembly (no external crates)
    let json = format!(
        "{{\n  \"source_path\": \"{}\",\n  \"dest_path\": \"{}\",\n  \"file_name\": \"{}\",\n  \"modified_unix\": {}\n}}\n",
        escape_json(source.to_string_lossy()),
        escape_json(dest.to_string_lossy()),
        escape_json(dest.file_name().unwrap_or_default().to_string_lossy()),
        ts
    );
    file.write_all(json.as_bytes())
}

fn escape_json<S: AsRef<str>>(s: S) -> String {
    let mut out = String::new();
    for c in s.as_ref().chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push('?'),
            other => out.push(other),
        }
    }
    out
}

fn main() -> io::Result<()> {
    let cfg = parse_args();

    if !cfg.source.exists() {
        eprintln!("Source directory not found: {}", cfg.source.display());
        std::process::exit(1);
    }

    let latest = match latest_file(&cfg.source)? {
        Some(p) => p,
        None => {
            eprintln!("No candidate files found in {}", cfg.source.display());
            std::process::exit(1);
        }
    };

    let file_name = latest.file_name().unwrap().to_owned();
    fs::create_dir_all(&cfg.dest)?;
    let dest_path = cfg.dest.join(file_name);
    fs::copy(&latest, &dest_path)?;
    let mtime = fs::metadata(&latest)?.modified().unwrap_or(SystemTime::UNIX_EPOCH);

    // Write manifest
    write_manifest(&cfg.manifest, &latest, &dest_path, mtime)?;

    println!("Pulled latest candidate:\n  from: {}\n    to: {}\n",
        latest.display(),
        dest_path.display());

    Ok(())
}

