use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut input = PathBuf::from("codex_on_QA/out/qastructure_features.txt");
    let mut output = PathBuf::from("codex_on_QA/out/experiment_plan.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "--input" => { if i+1 < args.len() { input = PathBuf::from(&args[i+1]); i+=1; } }
            "--out" | "--output" => { if i+1 < args.len() { output = PathBuf::from(&args[i+1]); i+=1; } }
            "-h" | "--help" => { print_usage(); return Ok(()); }
            _ => {}
        }
        i += 1;
    }

    let mut text = String::new();
    File::open(&input)?.read_to_string(&mut text)?;

    let has_moons = contains_any(&text, &["Moons", "two moons", "moons"]);
    let has_swiss = contains_any(&text, &["Swiss roll", "Swiss", "swiss roll"]);
    let has_louvain = contains_any(&text, &["Louvain", "louvain"]);
    let has_leiden = contains_any(&text, &["Leiden", "leiden"]);
    let has_spectral = contains_any(&text, &["Spectral", "Fiedler", "Laplacian"]);
    let has_infomap = contains_any(&text, &["Infomap", "infomap"]);
    let has_walktrap = contains_any(&text, &["Walktrap", "walktrap"]);

    // Default real graph dataset available in repo (fallback if football.gml absent)
    let graph_path = detect_graphml_path();

    let mut plan_items: Vec<String> = Vec::new();

    if has_moons { plan_items.push("synthetic_moons".to_string()); }
    if has_swiss { plan_items.push("synthetic_swiss_roll".to_string()); }
    if has_spectral { plan_items.push("graph_spectral".to_string()); }
    if has_louvain { plan_items.push("graph_louvain_like".to_string()); }
    if has_leiden { plan_items.push("graph_leiden_like".to_string()); }
    if has_infomap { plan_items.push("graph_infomap_like".to_string()); }
    if has_walktrap { plan_items.push("graph_walktrap_like".to_string()); }

    // Compose a JSON plan manually (no serde dependency)
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"source_outline\": \"{}\",\n", escape(&input.to_string_lossy())));
    json.push_str("  \"datasets\": {\n");
    json.push_str(&format!("    \"graph\": \"{}\"\n", escape(&graph_path)));
    json.push_str("  },\n");
    json.push_str("  \"experiments\": [\n");
    for (idx, item) in plan_items.iter().enumerate() {
        let comma = if idx + 1 == plan_items.len() { "" } else { "," };
        json.push_str(&format!("    \"{}\"{}\n", item, comma));
    }
    json.push_str("  ],\n");
    json.push_str("  \"qa_invariants\": [\n");
    json.push_str("    \"J=b*d\", \"X=e*d\", \"G=e^2+d^2\"\n");
    json.push_str("  ],\n");
    json.push_str("  \"notes\": \"Plan auto-extracted from outline; graph dataset defaults to qa_knowledge_graph.graphml if football.gml is absent.\"\n");
    json.push_str("}\n");

    if let Some(parent) = output.parent() { fs::create_dir_all(parent)?; }
    let mut f = File::create(&output)?;
    f.write_all(json.as_bytes())?;

    println!("Wrote plan to {}\nGraph dataset: {}", output.display(), graph_path);
    Ok(())
}

fn print_usage() {
    eprintln!("outline_to_plan --input codex_on_QA/out/qastructure_features.txt --output codex_on_QA/out/experiment_plan.json");
}

fn contains_any(text: &str, needles: &[&str]) -> bool { needles.iter().any(|n| text.contains(n)) }

fn detect_graphml_path() -> String {
    let candidates = [
        "qa_knowledge_graph.graphml",
        "artifacts/knowledge/qa_knowledge_graph.graphml",
        "qa_lab/qa_core/tests/football.gml", // if ever added
    ];
    for p in candidates {
        if std::path::Path::new(p).exists() { return p.to_string(); }
    }
    // Fallback to root graphml path string anyway
    candidates[0].to_string()
}

fn escape<S: AsRef<str>>(s: S) -> String {
    let mut out = String::new();
    for c in s.as_ref().chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            x if x.is_control() => out.push(' '),
            x => out.push(x),
        }
    }
    out
}

