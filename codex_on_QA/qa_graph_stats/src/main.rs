use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Default, Clone)]
struct NodeAttr { b: Option<f64>, e: Option<f64>, d: Option<f64>, a: Option<f64> }

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut graph_path = PathBuf::from("qa_knowledge_graph.graphml");
    let mut out_path = PathBuf::from("codex_on_QA/out/graph_stats.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph" => { if i+1 < args.len() { graph_path = PathBuf::from(&args[i+1]); i+=1; } }
            "--out" => { if i+1 < args.len() { out_path = PathBuf::from(&args[i+1]); i+=1; } }
            "-h" | "--help" => { print_usage(); return Ok(()); }
            _ => {}
        }
        i += 1;
    }

    let (nodes, edges, attrs) = parse_graphml(&graph_path)?;
    let n = nodes.len();
    let m = edges.len();
    let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (u, v) in &edges {
        adj.entry(*u).or_default().insert(*v);
        adj.entry(*v).or_default().insert(*u); // treat as undirected for stats
    }

    let mut degs: Vec<usize> = vec![0; n];
    for (i, _) in nodes.iter().enumerate() { degs[i] = adj.get(&i).map(|s| s.len()).unwrap_or(0); }

    // Compute clustering coefficient (local) using naive triangle counting per node
    let mut cc: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        let nei = match adj.get(&i) { Some(s) => s, None => { cc[i]=0.0; continue; } };
        let k = nei.len();
        if k < 2 { cc[i] = 0.0; continue; }
        let mut links = 0usize;
        let vec_nei: Vec<usize> = nei.iter().cloned().collect();
        for a in 0..vec_nei.len() {
            for b in (a+1)..vec_nei.len() {
                let u = vec_nei[a];
                let v = vec_nei[b];
                if adj.get(&u).map(|s| s.contains(&v)).unwrap_or(false) { links += 1; }
            }
        }
        let denom = k * (k - 1) / 2;
        cc[i] = if denom > 0 { links as f64 / denom as f64 } else { 0.0 };
    }

    // Harmonic centrality: sum_j 1/dist(i,j)
    let mut harm: Vec<f64> = vec![0.0; n];
    for i in 0..n { harm[i] = harmonic_centrality(i, &adj, n); }

    // Build invariants where possible
    let mut j_vals: Vec<f64> = Vec::with_capacity(n);
    let mut x_vals: Vec<f64> = Vec::with_capacity(n);
    let mut g_vals: Vec<f64> = Vec::with_capacity(n);
    let mut top_j: Vec<(String, f64)> = Vec::new();
    let mut top_x: Vec<(String, f64)> = Vec::new();
    let mut top_g: Vec<(String, f64)> = Vec::new();

    for (idx, name) in nodes.iter().enumerate() {
        let b = attrs.get(name).and_then(|a| a.b).unwrap_or(degs[idx] as f64);
        let e = attrs.get(name).and_then(|a| a.e).unwrap_or(cc[idx]);
        let d = attrs.get(name).and_then(|a| a.d).unwrap_or(harm[idx]);
        let j = b * d;
        let x = e * d;
        let g = e*e + d*d;
        j_vals.push(j); x_vals.push(x); g_vals.push(g);
        maybe_insert_top(&mut top_j, name.clone(), j);
        maybe_insert_top(&mut top_x, name.clone(), x);
        maybe_insert_top(&mut top_g, name.clone(), g);
    }

    // Compose JSON output
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"graph\": \"{}\",\n", escape(&graph_path.to_string_lossy())));
    json.push_str(&format!("  \"nodes\": {}, \"edges\": {},\n", n, m));
    json.push_str(&format!("  \"avg_degree\": {:.4},\n", (2.0 * m as f64) / (n as f64)));
    json.push_str("  \"invariants\": {\n");
    json.push_str(&format!(
        "    \"J\": {{ \"mean\": {:.6}, \"max\": {:.6} }},\n",
        mean(&j_vals), max(&j_vals)
    ));
    json.push_str(&format!(
        "    \"X\": {{ \"mean\": {:.6}, \"max\": {:.6} }},\n",
        mean(&x_vals), max(&x_vals)
    ));
    json.push_str(&format!(
        "    \"G\": {{ \"mean\": {:.6}, \"max\": {:.6} }}\n",
        mean(&g_vals), max(&g_vals)
    ));
    json.push_str("  },\n");
    json.push_str("  \"top_nodes\": {\n");
    json.push_str(&format!("    \"by_J\": {},\n", top_list_json(&top_j)));
    json.push_str(&format!("    \"by_X\": {},\n", top_list_json(&top_x)));
    json.push_str(&format!("    \"by_G\": {}\n", top_list_json(&top_g)));
    json.push_str("  }\n");
    json.push_str("}\n");

    if let Some(p) = out_path.parent() { fs::create_dir_all(p)?; }
    let mut f = File::create(&out_path)?;
    f.write_all(json.as_bytes())?;
    println!("Wrote stats to {}", out_path.display());
    Ok(())
}

fn print_usage() {
    eprintln!("qa_graph_stats --graph qa_knowledge_graph.graphml --out codex_on_QA/out/graph_stats.json");
}

fn parse_graphml(path: &PathBuf) -> io::Result<(Vec<String>, Vec<(usize,usize)>, HashMap<String, NodeAttr>)> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut nodes: Vec<String> = Vec::new();
    let mut node_index: HashMap<String, usize> = HashMap::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut attrs: HashMap<String, NodeAttr> = HashMap::new();

    let mut current_node: Option<String> = None;
    let mut current_key: Option<String> = None;
    // Map GraphML data keys to attr names by observation of provided file
    // d1=b, d2=e, d3=d, d4=a (from leading keys in the file)
    for line_res in reader.lines() {
        let line = line_res?;
        let s = line.trim();
        if s.starts_with("<node ") {
            if let Some(id) = extract_attr(s, "id") {
                current_node = Some(id.clone());
                if !node_index.contains_key(&id) {
                    let idx = nodes.len();
                    nodes.push(id.clone());
                    node_index.insert(id.clone(), idx);
                    attrs.entry(id.clone()).or_default();
                }
            }
        } else if s.starts_with("</node>") {
            current_node = None;
        } else if s.starts_with("<data ") {
            current_key = extract_attr(s, "key");
            // inline content?
            if let Some(val) = extract_data_inline(s) {
                if let (Some(node), Some(key)) = (current_node.clone(), current_key.clone()) {
                    apply_attr(&mut attrs, &node, &key, &val);
                }
                current_key = None;
            }
        } else if s.starts_with("</data>") {
            current_key = None;
        } else if s.starts_with("<edge ") {
            let src = extract_attr(s, "source");
            let dst = extract_attr(s, "target");
            if let (Some(sid), Some(tid)) = (src, dst) {
                if let (Some(&u), Some(&v)) = (node_index.get(&sid), node_index.get(&tid)) {
                    edges.push((u, v));
                }
            }
        } else if let (Some(node), Some(key)) = (current_node.clone(), current_key.clone()) {
            // data content on separate line
            if s.starts_with("<") && s.ends_with(">") { /* ignore nested tags */ } else {
                apply_attr(&mut attrs, &node, &key, s);
            }
        }
    }

    Ok((nodes, edges, attrs))
}

fn apply_attr(map: &mut HashMap<String, NodeAttr>, node: &str, key: &str, val: &str) {
    let entry = map.entry(node.to_string()).or_default();
    let v = val.trim();
    match key {
        "d1" => { entry.b = parse_num(v); }
        "d2" => { entry.e = parse_num(v); }
        "d3" => { entry.d = parse_num(v); }
        "d4" => { entry.a = parse_num(v); }
        _ => {}
    }
}

fn parse_num(s: &str) -> Option<f64> {
    let cleaned = s.replace(",", ".");
    cleaned.parse::<f64>().ok()
}

fn extract_attr(s: &str, attr: &str) -> Option<String> {
    // find attr="value"
    let pat = format!("{}=\"", attr);
    if let Some(start) = s.find(&pat) {
        let rest = &s[start+pat.len()..];
        if let Some(end) = rest.find('"') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

fn extract_data_inline(s: &str) -> Option<String> {
    // <data key="d1">VALUE</data>
    if let Some(start) = s.find('>') {
        if let Some(end) = s.rfind("</data>") {
            if end > start+1 { return Some(s[start+1..end].to_string()); }
        }
    }
    None
}

fn harmonic_centrality(src: usize, adj: &HashMap<usize, HashSet<usize>>, n: usize) -> f64 {
    // BFS distances
    let mut dist: Vec<i32> = vec![-1; n];
    let mut q: Vec<usize> = Vec::new();
    dist[src] = 0; q.push(src);
    let mut head = 0usize;
    while head < q.len() {
        let u = q[head]; head += 1;
        if let Some(nei) = adj.get(&u) {
            for &v in nei {
                if dist[v] == -1 { dist[v] = dist[u] + 1; q.push(v); }
            }
        }
    }
    let mut h = 0.0;
    for (i, &d) in dist.iter().enumerate() {
        if i == src { continue; }
        if d > 0 { h += 1.0 / (d as f64); }
    }
    h
}

fn maybe_insert_top(list: &mut Vec<(String, f64)>, name: String, val: f64) {
    list.push((name, val));
    list.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    if list.len() > 10 { list.truncate(10); }
}

fn mean(v: &Vec<f64>) -> f64 { if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 } }
fn max(v: &Vec<f64>) -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }

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

fn top_list_json(list: &Vec<(String,f64)>) -> String {
    let mut s = String::new();
    s.push('[');
    for (i, (name, val)) in list.iter().enumerate() {
        if i > 0 { s.push_str(", "); }
        s.push_str(&format!("{{ \"node\": \"{}\", \"value\": {:.6} }}", escape(name), val));
    }
    s.push(']');
    s
}
