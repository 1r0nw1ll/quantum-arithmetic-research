use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Default, Clone)]
struct NodeAttr { b: Option<f64>, e: Option<f64>, d: Option<f64>, a: Option<f64> }

#[derive(Clone, Copy)]
enum Mode { Baseline, X, J, Mix(f64), Full, FullMulti, E8Embed }

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut graph_path = PathBuf::from("qa_knowledge_graph.graphml");
    let mut out_path = PathBuf::from("codex_on_QA/out/graph_spectral_summary.json");
    let mut alpha = 0.5f64;
    let mut k_list: Vec<usize> = vec![2,4,6,8,10];
    let mut tau_scale: f64 = 1.0;
    let mut hi_beta: f64 = 0.0; // 0 disables harmonicity weighting
    let mut hi_source = String::from("canonical"); // canonical | markovian (from t{n}_hi)
    let mut alphas: Option<Vec<f64>> = None; // per-tuple weights for full-multi
    let mut e8_plane = String::from("be"); // be | jx
    let mut phase_mode = String::from("sincos"); // none | raw | sincos
    let mut scale_mode = String::from("zscore"); // none | zscore
    let mut qa_mode = String::from("qa21"); // qa21 | qa27

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph" => { if i+1 < args.len() { graph_path = PathBuf::from(&args[i+1]); i+=1; } }
            "--out" => { if i+1 < args.len() { out_path = PathBuf::from(&args[i+1]); i+=1; } }
            "--alpha" => { if i+1 < args.len() { alpha = args[i+1].parse().unwrap_or(0.5); i+=1; } }
            "--k" => { if i+1 < args.len() { k_list = args[i+1].split(',').filter_map(|s| s.parse::<usize>().ok()).collect(); i+=1; } }
            "--tau_scale" => { if i+1 < args.len() { tau_scale = args[i+1].parse().unwrap_or(1.0); i+=1; } }
            "--hi-beta" => { if i+1 < args.len() { hi_beta = args[i+1].parse().unwrap_or(0.0); i+=1; } }
            "--hi-source" => { if i+1 < args.len() { hi_source = args[i+1].to_string(); i+=1; } }
            "--alphas" => { if i+1 < args.len() { let s = args[i+1].clone(); i+=1; let v: Vec<f64> = s.split(',').filter_map(|t| t.parse::<f64>().ok()).collect(); if !v.is_empty() { alphas = Some(v); } } }
            "--phase-mode" => { if i+1 < args.len() { phase_mode = args[i+1].to_string(); i+=1; } }
            "--scale-mode" => { if i+1 < args.len() { scale_mode = args[i+1].to_string(); i+=1; } }
            "--qa-mode" => { if i+1 < args.len() { qa_mode = args[i+1].to_string(); i+=1; } }
            "--e8-plane" => { if i+1 < args.len() { e8_plane = args[i+1].to_string(); i+=1; } }
            "-h" | "--help" => { print_usage(); return Ok(()); }
            _ => {}
        }
        i += 1;
    }

    let t0 = Instant::now();
    let (names, edges, attrs, gt_raw, multi_be, multi_hi) = parse_graphml(&graph_path)?;
    let n = names.len();
    let mut base_adj: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    // Build undirected 0/1 adjacency
    for (u, v) in edges {
        if u == v { continue; }
        base_adj[u][v] = 1.0; base_adj[v][u] = 1.0;
    }
    let (deg0, cc0, harm0) = derive_graph_stats(&base_adj);
    let (j_vec, x_vec) = derive_qainvariants(&names, &attrs, &deg0, &cc0, &harm0);
    let (j_multi, x_multi) = derive_qainvariants_multi(&multi_be);
    let w_multi = compute_tuple_weights(&multi_be, &multi_hi, hi_beta, &hi_source);

    // Prepare ground-truth labels (if present)
    let gt_labels = compress_gt(&gt_raw);

    // Run experiments for baseline and QA-weighted modes
    let modes = vec![Mode::Baseline, Mode::X, Mode::J, Mode::Mix(alpha), Mode::Full, Mode::FullMulti, Mode::E8Embed];
    let mut results_json = String::new();
    results_json.push_str("{\n");
    results_json.push_str(&format!("  \"graph\": \"{}\",\n", escape(&graph_path.to_string_lossy())));
    results_json.push_str(&format!("  \"nodes\": {},\n", n));
    results_json.push_str(&format!("  \"k_candidates\": \"{}\",\n", k_list.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",")));
    results_json.push_str(&format!("  \"alpha\": {:.3},\n", alpha));
    results_json.push_str(&format!("  \"qa_mode\": \"{}\",\n", escape(&qa_mode)));
    // Multi-tuple kernel diagnostics
    let alphas_str = match &alphas { Some(v) => v.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(","), None => String::new() };
    results_json.push_str(&format!("  \"hi_beta\": {:.6},\n", hi_beta));
    results_json.push_str(&format!("  \"hi_source\": \"{}\",\n", escape(&hi_source)));
    results_json.push_str(&format!("  \"alphas\": \"{}\",\n", escape(&alphas_str)));
    results_json.push_str(&format!("  \"elapsed_precompute_ms\": {} ,\n", t0.elapsed().as_millis()));
    results_json.push_str("  \"runs\": [\n");
    // Precompute baseline Q as guard for tau selection
    let (_bk0, baseline_q, _bs0, _bl0) = try_spectral(&base_adj, &k_list);

    for (mi, mode) in modes.iter().enumerate() {
        let w = match *mode {
            Mode::Full => {
                let phi = compute_full_invariants_single(&names, &attrs, &deg0, &cc0, &harm0, qa_mode.as_str());
                let phi_z = zscore_features(&phi);
                let tau = estimate_tau(&phi_z) * tau_scale.max(1e-6);
                build_kernel_adjacency_masked(&phi_z, tau, &base_adj)
            }
            Mode::FullMulti => {
                let phi = compute_full_invariants_multi(&multi_be, qa_mode.as_str(), alphas.as_ref());
                let phi_z = zscore_features(&phi);
                let tau = estimate_tau(&phi_z) * tau_scale.max(1e-6);
                build_kernel_adjacency_masked(&phi_z, tau, &base_adj)
            }
            Mode::E8Embed => {
                let phi = compute_e8_embed(&multi_be, &e8_plane);
                let phi_z = zscore_features(&phi);
                let tau = estimate_tau(&phi_z) * tau_scale.max(1e-6);
                build_kernel_adjacency_masked(&phi_z, tau, &base_adj)
            }
            _ => build_weighted_v2(&base_adj, &j_vec, &x_vec, Some(&j_multi), Some(&x_multi), Some(&w_multi), *mode),
        };
        let (best_k, best_q, best_sizes, best_labels) = if matches!(mode, Mode::Full) || matches!(mode, Mode::FullMulti) || matches!(mode, Mode::E8Embed) {
            // Build canonical phi, apply phase handling and scaling
            let mut phi = if matches!(mode, Mode::Full) {
                let mut p = compute_full_invariants_single(&names, &attrs, &deg0, &cc0, &harm0, qa_mode.as_str());
                p = apply_phase_mode(&p, phase_mode.as_str());
                p
            } else {
                if matches!(mode, Mode::FullMulti) {
                    compute_full_invariants_multi(&multi_be, qa_mode.as_str(), alphas.as_ref())
                } else {
                    compute_e8_embed(&multi_be, &e8_plane)
                }
            };
            let phi_scaled = match scale_mode.as_str() { "zscore" => zscore_features(&phi), _ => phi };
            let tau0 = estimate_tau(&phi_scaled).max(1e-6);
            let sweep_scales = vec![0.01, 0.05, 0.10, 0.20, 0.50];
            let (_tau_sel, k_sel, q_sel, sizes_sel, labels_sel) = select_best_tau(&phi_scaled, &base_adj, &names, &k_list, &gt_raw, baseline_q, &sweep_scales, tau0);
            (k_sel, q_sel, sizes_sel, labels_sel)
        } else {
            try_spectral(&w, &k_list)
        };
        // Emit per-mode artifacts: labels CSV and cluster stats JSON
        if let Some(parent) = out_path.parent() { fs::create_dir_all(parent)?; }
        let mode_str = mode_name(*mode);
        let labels_path = out_path.parent().unwrap().join(format!("labels_{}.csv", mode_str));
        write_labels_csv(&labels_path, &names, &best_labels, &deg0, &cc0, &harm0, &j_vec, &x_vec)?;
        let stats_path = out_path.parent().unwrap().join(format!("clusters_{}.json", mode_str));
        write_cluster_stats_json(&stats_path, best_k, &best_labels, &deg0, &cc0, &harm0, &j_vec, &x_vec)?;
        // Optional metrics versus ground truth
        let (purity, ari, nmi) = if let Some(gt) = gt_labels.as_ref() {
            let (p, a, n) = clustering_metrics(&best_labels, gt, best_k);
            (Some(p), Some(a), Some(n))
        } else { (None, None, None) };

        let mut run = String::new();
        run.push_str(&format!(
            "    {{ \n      \"mode\": \"{}\", \n      \"best_k\": {}, \n      \"modularity_Q\": {:.6}, \n      \"cluster_sizes\": {}, \n      \"labels_sample\": {}, \n      \"purity\": {}, \n      \"ARI\": {}, \n      \"NMI\": {}",
            mode_name(*mode), best_k, best_q, json_list_usize(&best_sizes), json_list_prefix(&best_labels, 20),
            opt_f64_json(purity), opt_f64_json(ari), opt_f64_json(nmi)
        ));
        if matches!(mode, Mode::Full) || matches!(mode, Mode::FullMulti) || matches!(mode, Mode::E8Embed) {
            let mut phi = if matches!(mode, Mode::Full) {
                let mut p = compute_full_invariants_single(&names, &attrs, &deg0, &cc0, &harm0, qa_mode.as_str());
                p = apply_phase_mode(&p, phase_mode.as_str());
                p
            } else {
                if matches!(mode, Mode::FullMulti) {
                    compute_full_invariants_multi(&multi_be, qa_mode.as_str(), alphas.as_ref())
                } else {
                    compute_e8_embed(&multi_be, &e8_plane)
                }
            };
            let phi_scaled = match scale_mode.as_str() { "zscore" => zscore_features(&phi), _ => phi };
            let tau0 = estimate_tau(&phi_scaled).max(1e-6);
            let sweep_scales = vec![0.01, 0.05, 0.10, 0.20, 0.50];
            let (tau_sel, k_sel, q_sel, _sizes_sel, _labels_sel) = select_best_tau(&phi_scaled, &base_adj, &names, &k_list, &gt_raw, baseline_q, &sweep_scales, tau0);
            run.push_str(&format!(
                ",\n      \"phase_mode\": \"{}\", \n      \"scale_mode\": \"{}\", \n      \"tau_selected\": {:.6}, \n      \"selected_metrics\": {{ \"k\": {}, \"Q\": {:.6} }},\n      \"tau_sweep\": {{\n",
                phase_mode, scale_mode, tau_sel, k_sel, q_sel
            ));
            for (si, s) in sweep_scales.iter().enumerate() {
                let tau = tau0 * *s;
                let w_s = build_kernel_adjacency_masked(&phi_scaled, tau, &base_adj);
                let (k_s, q_s, _sizes_s, labels_s) = try_spectral(&w_s, &k_list);
                let (pur_s, ari_s, nmi_s) = if let Some(gt) = gt_labels.as_ref() { clustering_metrics(&labels_s, gt, k_s) } else { (0.0,0.0,0.0) };
                run.push_str(&format!(
                    "        \"{:.2}\": {{ \"best_k\": {}, \"Q\": {:.6}, \"Purity\": {:.6}, \"ARI\": {:.6}, \"NMI\": {:.6} }}{}\n",
                    s, k_s, q_s, pur_s, ari_s, nmi_s, if si+1==sweep_scales.len(){""}else{","}
                ));
            }
            run.push_str("      }\n");
        }
        run.push_str(&format!("\n    }}{}\n", if mi+1 == modes.len() { "" } else { "," }));
        results_json.push_str(&run);
    }

    results_json.push_str("  ]\n}\n");
    if let Some(p) = out_path.parent() { fs::create_dir_all(p)?; }
    let mut f = File::create(&out_path)?;
    f.write_all(results_json.as_bytes())?;
    println!("Wrote spectral summary to {}", out_path.display());
    Ok(())
}

fn print_usage() {
    eprintln!("qa_graph_experiments --graph qa_knowledge_graph.graphml --out codex_on_QA/out/graph_spectral_summary.json [--alpha 0.5] [--k 2,4,6,8,10] [--tau_scale 1.0] [--phase-mode none|raw|sincos] [--scale-mode none|zscore] [--qa-mode qa21|qa27]");
}

fn parse_graphml(path: &PathBuf) -> io::Result<(
    Vec<String>,
    Vec<(usize,usize)>,
    HashMap<String, NodeAttr>,
    Vec<Option<usize>>,
    Vec<Vec<(f64,f64)>>,
    Vec<Vec<f64>>, // optional HI per tuple (NaN if missing)
)> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut nodes: Vec<String> = Vec::new();
    let mut node_index: HashMap<String, usize> = HashMap::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut attrs: HashMap<String, NodeAttr> = HashMap::new();
    let mut gt: Vec<Option<usize>> = Vec::new();
    let mut key_to_name: HashMap<String, String> = HashMap::new();

    // Multi-tuple (b,e) per node collected as vector per node index
    let mut multi_be: Vec<Vec<(f64,f64)>> = Vec::new();
    let mut multi_hi: Vec<Vec<f64>> = Vec::new();
    // For the current node, collect partial (b,e) by tuple index n
    let mut curr_idx: Option<usize> = None;
    let mut curr_pairs: HashMap<usize, (Option<f64>, Option<f64>)> = HashMap::new();
    let mut curr_hi: HashMap<usize, Option<f64>> = HashMap::new();

    let mut current_node: Option<String> = None;
    let mut current_key: Option<String> = None;
    for line_res in reader.lines() {
        let line = line_res?;
        let s = line.trim();
        if s.starts_with("<key ") {
            if let (Some(id), Some(aname), Some(forwho)) = (extract_attr(s, "id"), extract_attr(s, "attr.name"), extract_attr(s, "for")) {
                if forwho == "node" { key_to_name.insert(id, aname); }
            }
        } else if s.starts_with("<node ") {
            if let Some(id) = extract_attr(s, "id") {
                current_node = Some(id.clone());
                if !node_index.contains_key(&id) {
                    let idx = nodes.len();
                    nodes.push(id.clone());
                    node_index.insert(id.clone(), idx);
                    attrs.entry(id.clone()).or_default();
                    gt.push(None);
                    multi_be.push(Vec::new());
                    multi_hi.push(Vec::new());
                }
                // reset per-node collector
                curr_idx = node_index.get(&id).cloned();
                curr_pairs.clear();
                curr_hi.clear();
            }
        } else if s.starts_with("</node>") {
            // finalize multi tuples for this node
            if let Some(i) = curr_idx {
                let mut pairs: Vec<(usize, (Option<f64>, Option<f64>))> = curr_pairs.iter().map(|(k,v)| (*k, *v)).collect();
                pairs.sort_by_key(|(k, _)| *k);
                let mut out: Vec<(f64,f64)> = Vec::new();
                let mut hi_out: Vec<f64> = Vec::new();
                for (_k, (b_opt, e_opt)) in pairs {
                    if let (Some(b), Some(e)) = (b_opt, e_opt) {
                        out.push((b,e));
                        let h = curr_hi.get(&_k).copied().flatten().unwrap_or(std::f64::NAN);
                        hi_out.push(h);
                    }
                }
                multi_be[i] = out;
                multi_hi[i] = hi_out;
            }
            current_node = None;
            curr_idx = None;
            curr_pairs.clear();
            curr_hi.clear();
        } else if s.starts_with("<data ") {
            current_key = extract_attr(s, "key");
            if let Some(val) = extract_data_inline(s) {
                if let (Some(node), Some(key)) = (current_node.clone(), current_key.clone()) {
                    apply_attr(&mut attrs, &node, &key, &val);
                    if let Some(aname) = key_to_name.get(&key) {
                        if aname == "value" {
                            if let Some(&idx) = node_index.get(&node) {
                                gt[idx] = parse_usize(val.trim());
                            }
                        } else if aname.starts_with("t") {
                            // expect t{n}_b or t{n}_e
                            if let Some(rest) = aname.strip_prefix('t') {
                                if let Some(pos) = rest.find('_') {
                                    let (nstr, kind) = (&rest[..pos], &rest[pos+1..]);
                                    if let Ok(n) = nstr.parse::<usize>() {
                                        if let Some(i) = curr_idx {
                                            let entry = curr_pairs.entry(n).or_insert((None,None));
                                            if kind == "b" { entry.0 = parse_num(val.trim()); }
                                            if kind == "e" { entry.1 = parse_num(val.trim()); }
                                            if kind == "hi" { curr_hi.insert(n, parse_num(val.trim())); }
                                        }
                                    }
                                }
                            }
                        }
                    }
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
            if s.starts_with("<") && s.ends_with(">") { /* ignore */ } else { apply_attr(&mut attrs, &node, &key, s); }
        }
    }
    Ok((nodes, edges, attrs, gt, multi_be, multi_hi))
}

fn apply_attr(map: &mut HashMap<String, NodeAttr>, node: &str, key: &str, val: &str) {
    let entry = map.entry(node.to_string()).or_default();
    let v = val.trim();
    match key { "d1" => entry.b = parse_num(v), "d2" => entry.e = parse_num(v), "d3" => entry.d = parse_num(v), "d4" => entry.a = parse_num(v), _ => {} }
}
fn parse_num(s: &str) -> Option<f64> { s.replace(",", ".").parse::<f64>().ok() }
fn parse_usize(s: &str) -> Option<usize> { s.parse::<usize>().ok() }
fn extract_attr(s: &str, attr: &str) -> Option<String> { let pat = format!("{}=\"", attr); s.find(&pat).and_then(|i| { let r=&s[i+pat.len()..]; r.find('"').map(|j| r[..j].to_string()) }) }
fn extract_data_inline(s: &str) -> Option<String> { if let Some(i)=s.find('>'){ if let Some(j)=s.rfind("</data>"){ if j>i+1 { return Some(s[i+1..j].to_string()); }}} None }

fn derive_graph_stats(adj: &Vec<Vec<f64>>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = adj.len();
    let mut deg: Vec<f64> = vec![0.0; n];
    for i in 0..n { deg[i] = adj[i].iter().sum::<f64>(); }
    let mut cc: Vec<f64> = vec![0.0; n];
    for i in 0..n { cc[i] = clustering_coeff(i, adj); }
    let mut harm: Vec<f64> = vec![0.0; n];
    for i in 0..n { harm[i] = harmonic_centrality(i, adj); }
    (deg, cc, harm)
}

fn clustering_coeff(i: usize, adj: &Vec<Vec<f64>>) -> f64 {
    let n = adj.len();
    let mut nei: Vec<usize> = Vec::new();
    for j in 0..n { if adj[i][j] > 0.0 { nei.push(j); } }
    let k = nei.len(); if k < 2 { return 0.0; }
    let mut links = 0usize;
    for a in 0..k { for b in (a+1)..k { let u=nei[a]; let v=nei[b]; if adj[u][v] > 0.0 { links += 1; } } }
    let denom = k * (k-1) / 2; if denom>0 { links as f64 / denom as f64 } else { 0.0 }
}

fn harmonic_centrality(src: usize, adj: &Vec<Vec<f64>>) -> f64 {
    let n = adj.len();
    let mut dist: Vec<i32> = vec![-1; n];
    let mut q: Vec<usize> = Vec::new();
    dist[src]=0; q.push(src);
    let mut head=0usize;
    while head<q.len() { let u=q[head]; head+=1; for v in 0..n { if adj[u][v] > 0.0 && dist[v]==-1 { dist[v]=dist[u]+1; q.push(v); } } }
    let mut h = 0.0; for i in 0..n { if i!=src { let d=dist[i]; if d>0 { h += 1.0/(d as f64); } } } h
}

fn derive_qainvariants(names: &Vec<String>, attrs: &HashMap<String, NodeAttr>, deg: &Vec<f64>, cc: &Vec<f64>, harm: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = names.len();
    let mut jv = vec![0.0; n];
    let mut xv = vec![0.0; n];
    for i in 0..n {
        let name = &names[i];
        let a = attrs.get(name);
        let b = a.and_then(|x| x.b).unwrap_or(deg[i]);
        let e = a.and_then(|x| x.e).unwrap_or(cc[i]);
        let d = a.and_then(|x| x.d).unwrap_or(harm[i]);
        let j = b*d; let x = e*d;
        jv[i]=j; xv[i]=x;
    }
    (jv, xv)
}

fn derive_qainvariants_multi(multi_be: &Vec<Vec<(f64,f64)>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut jv: Vec<Vec<f64>> = Vec::with_capacity(multi_be.len());
    let mut xv: Vec<Vec<f64>> = Vec::with_capacity(multi_be.len());
    for pairs in multi_be.iter() {
        let mut j_row: Vec<f64> = Vec::with_capacity(pairs.len());
        let mut x_row: Vec<f64> = Vec::with_capacity(pairs.len());
        for (b, e) in pairs.iter() {
            let d = b + e;
            j_row.push(b * d);
            x_row.push(e * d);
        }
        jv.push(j_row);
        xv.push(x_row);
    }
    (jv, xv)
}

fn compute_tuple_weights(multi_be: &Vec<Vec<(f64,f64)>>, multi_hi: &Vec<Vec<f64>>, beta: f64, source: &str) -> Vec<Vec<f64>> {
    // Weight per tuple based on either canonical dev or provided markovian HI (t{n}_hi):
    // w = exp(-beta * hi). If beta==0, all weights = 1.
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(multi_be.len());
    for (row_idx, pairs) in multi_be.iter().enumerate() {
        let mut w_row: Vec<f64> = Vec::with_capacity(pairs.len());
        for (tidx, (b, e)) in pairs.iter().enumerate() {
            let hi = if source == "markovian" {
                // read from multi_hi if available, else fallback to canonical
                let v = multi_hi.get(row_idx).and_then(|r| r.get(tidx)).copied().unwrap_or(std::f64::NAN);
                if v.is_nan() {
                    // canonical fallback
                    let d = b + e; let a = b + 2.0*e; let c = 2.0*d*e; let f = b*a; let g = e*e + d*d;
                    ((g*g) - (c*c + f*f)).abs() / (1.0 + g*g + c*c + f*f)
                } else { v }
            } else {
                // canonical
                let d = b + e; let a = b + 2.0*e; let c = 2.0*d*e; let f = b*a; let g = e*e + d*d;
                ((g*g) - (c*c + f*f)).abs() / (1.0 + g*g + c*c + f*f)
            };
            let w = if beta <= 0.0 { 1.0 } else { (-beta * hi).exp() };
            w_row.push(w);
        }
        out.push(w_row);
    }
    out
}

fn build_weighted_v2(
    base_adj: &Vec<Vec<f64>>,
    j_single: &Vec<f64>,
    x_single: &Vec<f64>,
    j_multi: Option<&Vec<Vec<f64>>>,
    x_multi: Option<&Vec<Vec<f64>>>,
    w_multi: Option<&Vec<Vec<f64>>>,
    mode: Mode,
) -> Vec<Vec<f64>> {
    fn avg_abs_diff(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        let t = a.len().min(b.len());
        if t == 0 { return 0.0; }
        let mut s = 0.0; for i in 0..t { s += (a[i] - b[i]).abs(); }
        s / (t as f64)
    }
    fn wavg_abs_diff(a: &Vec<f64>, b: &Vec<f64>, wa: &Vec<f64>, wb: &Vec<f64>) -> f64 {
        let t = a.len().min(b.len()).min(wa.len()).min(wb.len());
        if t == 0 { return 0.0; }
        let mut num = 0.0; let mut den = 0.0;
        for i in 0..t {
            let w = 0.5*(wa[i] + wb[i]);
            num += w * (a[i] - b[i]).abs();
            den += w;
        }
        if den <= 0.0 { avg_abs_diff(a,b) } else { num/den }
    }

    let n = base_adj.len();
    let mut w = vec![vec![0.0; n]; n];
    let has_multi = match (&j_multi, &x_multi) {
        (Some(jm), Some(xm)) => !jm.is_empty() && !xm.is_empty(),
        _ => false,
    };
    for i in 0..n {
        for jdx in 0..n {
            if i==jdx { continue; }
            let a = base_adj[i][jdx]; if a <= 0.0 { continue; }
            let weight = match mode {
                Mode::Baseline => 1.0,
                Mode::X => {
                    if has_multi && j_multi.unwrap()[i].len()>0 && j_multi.unwrap()[jdx].len()>0 {
                        let dx = match w_multi {
                            Some(wm) => wavg_abs_diff(&x_multi.unwrap()[i], &x_multi.unwrap()[jdx], &wm[i], &wm[jdx]),
                            None => avg_abs_diff(&x_multi.unwrap()[i], &x_multi.unwrap()[jdx]),
                        };
                        1.0 / (1.0 + dx)
                    } else {
                        1.0 / (1.0 + (x_single[i]-x_single[jdx]).abs())
                    }
                }
                Mode::J => {
                    if has_multi && j_multi.unwrap()[i].len()>0 && j_multi.unwrap()[jdx].len()>0 {
                        let dj = match w_multi {
                            Some(wm) => wavg_abs_diff(&j_multi.unwrap()[i], &j_multi.unwrap()[jdx], &wm[i], &wm[jdx]),
                            None => avg_abs_diff(&j_multi.unwrap()[i], &j_multi.unwrap()[jdx]),
                        };
                        1.0 / (1.0 + dj)
                    } else {
                        1.0 / (1.0 + (j_single[i]-j_single[jdx]).abs())
                    }
                }
                Mode::Mix(alpha) => {
                    if has_multi && j_multi.unwrap()[i].len()>0 && j_multi.unwrap()[jdx].len()>0 {
                        let dx = match w_multi {
                            Some(wm) => wavg_abs_diff(&x_multi.unwrap()[i], &x_multi.unwrap()[jdx], &wm[i], &wm[jdx]),
                            None => avg_abs_diff(&x_multi.unwrap()[i], &x_multi.unwrap()[jdx]),
                        };
                        let dj = match w_multi {
                            Some(wm) => wavg_abs_diff(&j_multi.unwrap()[i], &j_multi.unwrap()[jdx], &wm[i], &wm[jdx]),
                            None => avg_abs_diff(&j_multi.unwrap()[i], &j_multi.unwrap()[jdx]),
                        };
                        1.0 / (1.0 + alpha*dx + (1.0-alpha)*dj)
                    } else {
                        1.0 / (1.0 + alpha*(x_single[i]-x_single[jdx]).abs() + (1.0-alpha)*(j_single[i]-j_single[jdx]).abs())
                    }
                }
                Mode::Full | Mode::FullMulti | Mode::E8Embed => unreachable!(),
            };
            w[i][jdx] = a * weight;
        }
    }
    for i in 0..n { for jdx in (i+1)..n { let v = 0.5*(w[i][jdx]+w[jdx][i]); w[i][jdx]=v; w[jdx][i]=v; } }
    w
}

// --- Canonical QA invariants (aligned with QA_CANONICAL_INVARIANTS.md) ---
#[derive(Clone, Debug)]
struct QaNodeFeatures {
    // Core tuple and squares
    b: f64, e: f64, d: f64, a: f64,
    bb: f64, ee: f64, dd: f64, aa: f64,
    // Triangle sides
    c: f64, f: f64, g: f64,
    // Triangle/ellipse composites
    l: f64, h_sum: f64, i_diff: f64,
    // Primary invariants
    j: f64, x: f64, k: f64,
    // Secondary invariants
    w: f64, y: f64, z: f64,
    // Ellipse-related extras
    h_inner: f64,        // sqrt(F) = sqrt(b*a) (inner ellipse minor semi-axis)
    h_quantum: f64,      // d * sqrt(F) (semi minor measure for quantum ellipse)
    // Phases (raw)
    phase24: f64, phase9: f64, phase72: f64, phase24_bin: f64,
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps * (1.0 + a.abs() + b.abs())
}

fn build_canonical_features(b: f64, e: f64, d: f64, a: f64) -> QaNodeFeatures {
    let bb = b * b;
    let ee = e * e;
    let dd = d * d;
    let aa = a * a;
    // Triangle sides
    let x = e * d;         // X
    let c = 2.0 * x;       // C = 2X = 2ed
    let f = b * a;         // F = ba
    let g = ee + dd;       // G = e^2 + d^2
    // Composites
    let l = (c * f) / 12.0;    // L = CF/12
    let h_sum = c + f;         // H = C + F
    let i_diff = (c - f).abs();// I = |C-F|
    // Primary
    let j = b * d;         // J = bd
    let k = d * a;         // K = da
    // Secondary (canonical)
    let w = x + k;         // W = X + K = d(e+a)
    let y = aa - dd;       // Y = A - D
    let z = ee + k;        // Z = E + K
    // Ellipse measures
    let h_inner = (f).abs().sqrt();     // sqrt(F)
    let h_quantum = d * h_inner;        // d * sqrt(F)
    // Phases
    let phase24 = modulo(k, 24.0);
    let phase9  = modulo(k, 9.0);
    let phase72 = modulo(k, 72.0);
    let phase24_bin = phase24.floor().min(23.0);

    // Closure checks (debug only)
    if cfg!(debug_assertions) {
        debug_assert!(approx_eq(c * c + f * f, g * g, 1e-9), "Pythagorean closure failed");
        debug_assert!(approx_eq(w, x + k, 1e-9), "W != X + K closure failed");
        debug_assert!(approx_eq(y, aa - dd, 1e-9), "Y != A - D closure failed");
        debug_assert!(approx_eq(z, ee + k, 1e-9), "Z != E + K closure failed");
    }

    QaNodeFeatures { b, e, d, a, bb, ee, dd, aa, c, f, g, l, h_sum, i_diff, j, x, k, w, y, z,
                     h_inner, h_quantum, phase24, phase9, phase72, phase24_bin }
}

fn vector_for_qa_mode(f: &QaNodeFeatures, qa_mode: &str) -> Vec<f64> {
    // Build core vector for qa_mode (without phases). Append phases later.
    match qa_mode {
        "qa21" => vec![
            // Canonical 21: b,e,d,a,B,E,D,A,X,C,F,G,L,H,I,J,K,W,Y,Z,h
            f.b, f.e, f.d, f.a,
            f.bb, f.ee, f.dd, f.aa,
            f.x, f.c, f.f, f.g,
            f.l, f.h_sum, f.i_diff,
            f.j, f.k, f.w, f.y, f.z,
            f.h_quantum,
        ],
        "qa27" => {
            let eps = if f.a != 0.0 { f.e / f.a } else { 0.0 };
            let f_over_c = if f.c != 0.0 { f.f / f.c } else { 0.0 };
            let g_over_c = if f.c != 0.0 { f.g / f.c } else { 0.0 };
            let r_h = (f.bb + f.ee + f.dd + f.aa).sqrt();
            let e_qa = f.g * f.g - f.c * f.f;
            let theta = f.e.atan2(f.b);
            let mut v = vec![
                // qa21 block
                f.b, f.e, f.d, f.a,
                f.bb, f.ee, f.dd, f.aa,
                f.x, f.c, f.f, f.g,
                f.l, f.h_sum, f.i_diff,
                f.j, f.k, f.w, f.y, f.z,
                f.h_quantum,
            ];
            v.extend_from_slice(&[eps, f_over_c, g_over_c, r_h, e_qa, theta]);
            v
        }
        _ => {
            // Default to qa21
            vec![
                f.b, f.e, f.d, f.a,
                f.bb, f.ee, f.dd, f.aa,
                f.x, f.c, f.f, f.g,
                f.l, f.h_sum, f.i_diff,
                f.j, f.k, f.w, f.y, f.z,
                f.h_quantum,
            ]
        }
    }
}

// Build canonical feature vectors per node, falling back to (deg,cc,harm) if b,e,d absent.
fn compute_full_invariants_single(names: &Vec<String>, attrs: &HashMap<String, NodeAttr>, deg: &Vec<f64>, cc: &Vec<f64>, harm: &Vec<f64>, qa_mode: &str) -> Vec<Vec<f64>> {
    let n = names.len();
    let mut phi: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let name = &names[i];
        let at = attrs.get(name);
        let b = at.and_then(|x| x.b).unwrap_or(deg[i]);
        let e = at.and_then(|x| x.e).unwrap_or(cc[i]);
        let d = at.and_then(|x| x.d).unwrap_or(harm[i]);
        let a = at.and_then(|x| x.a).unwrap_or(d + e);
        let feats = build_canonical_features(b, e, d, a);
        let mut v = vector_for_qa_mode(&feats, qa_mode);
        // Append raw phases for downstream phase-mode handling
        v.extend_from_slice(&[feats.phase24, feats.phase9, feats.phase72, feats.phase24_bin]);
        phi.push(v);
    }
    phi
}

fn compute_full_invariants_multi(multi_be: &Vec<Vec<(f64,f64)>>, qa_mode: &str, alphas: Option<&Vec<f64>>) -> Vec<Vec<f64>> {
    // Build per-tuple feature vectors with per-tuple sin/cos phases, then per-tuple z-score across nodes,
    // and finally apply per-tuple alpha weights before concatenation.
    let n = multi_be.len();
    if n == 0 { return vec![]; }
    let t_max = multi_be.iter().map(|p| p.len()).max().unwrap_or(0);
    if t_max == 0 { return vec![vec![]; n]; }

    // Determine tuple feature length from a sample
    let (sb, se) = if !multi_be[0].is_empty() { (multi_be[0][0].0, multi_be[0][0].1) } else { (0.0, 0.0) };
    let d0 = sb + se; let a0 = sb + 2.0*se;
    let feats0 = build_canonical_features(sb, se, d0, a0);
    let base_len = vector_for_qa_mode(&feats0, qa_mode).len();
    let tuple_len = base_len + 2; // add sin/cos per tuple

    // Collect raw per-tuple features
    let mut per_tuple_feats: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; t_max]; n];
    for (i, pairs) in multi_be.iter().enumerate() {
        for (tidx, (b, e)) in pairs.iter().enumerate() {
            let d = b + e; let a = b + 2.0*e;
            let feats = build_canonical_features(*b, *e, d, a);
            let mut v = vector_for_qa_mode(&feats, qa_mode);
            // append per-tuple phase (sin/cos of theta = atan2(e,b))
            let theta = e.atan2(*b);
            v.push(theta.sin()); v.push(theta.cos());
            per_tuple_feats[i][tidx] = v;
        }
    }

    // Compute per-tuple means/stds across nodes for z-score
    let mut means: Vec<Vec<f64>> = vec![vec![0.0; tuple_len]; t_max];
    let mut stds: Vec<Vec<f64>> = vec![vec![0.0; tuple_len]; t_max];
    // means
    for tidx in 0..t_max {
        let mut count = 0.0;
        for i in 0..n {
            if per_tuple_feats[i].len() > tidx && !per_tuple_feats[i][tidx].is_empty() {
                let v = &per_tuple_feats[i][tidx];
                for d in 0..tuple_len { means[tidx][d] += v[d]; }
                count += 1.0;
            }
        }
        if count > 0.0 { for d in 0..tuple_len { means[tidx][d] /= count; } }
    }
    // stds
    for tidx in 0..t_max {
        let mut count = 0.0;
        for i in 0..n {
            if per_tuple_feats[i].len() > tidx && !per_tuple_feats[i][tidx].is_empty() {
                let v = &per_tuple_feats[i][tidx];
                for d in 0..tuple_len { let z = v[d] - means[tidx][d]; stds[tidx][d] += z*z; }
                count += 1.0;
            }
        }
        if count > 0.0 { for d in 0..tuple_len { stds[tidx][d] = (stds[tidx][d]/count).sqrt(); if stds[tidx][d]==0.0 { stds[tidx][d]=1.0; } } }
    }

    // Default alphas if not provided
    let default_alpha = |tidx: usize| -> f64 { match tidx { 0 => 1.0, 1 => 0.6, _ => 0.4 } };

    // Build final concatenated vectors
    let mut out_phi: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut vcat: Vec<f64> = Vec::with_capacity(t_max * tuple_len);
        for tidx in 0..t_max {
            if per_tuple_feats[i].len() > tidx && !per_tuple_feats[i][tidx].is_empty() {
                let v = &per_tuple_feats[i][tidx];
                let mut block: Vec<f64> = vec![0.0; tuple_len];
                for d in 0..tuple_len { block[d] = (v[d] - means[tidx][d]) / stds[tidx][d]; }
                let a = alphas.and_then(|vv| vv.get(tidx).copied()).unwrap_or_else(|| default_alpha(tidx));
                for d in 0..tuple_len { block[d] *= a; }
                vcat.extend_from_slice(&block);
            } else {
                // pad zeros for missing tuple
                vcat.extend(std::iter::repeat(0.0).take(tuple_len));
            }
        }
        out_phi.push(vcat);
    }
    out_phi
}

fn compute_e8_embed(multi_be: &Vec<Vec<(f64,f64)>>, e8_plane: &str) -> Vec<Vec<f64>> {
    // Build 8D embedding from 4 tuples: either (b,e) or (J,X) per tuple.
    // Vector: [p1,q1, p2,q2, p3,q3, p4,q4], 0 if missing.
    let n = multi_be.len();
    let t_max = 4usize;
    let mut phi: Vec<Vec<f64>> = Vec::with_capacity(n);
    for pairs in multi_be.iter() {
        let mut v = vec![0.0f64; t_max*2];
        for (tidx, (b,e)) in pairs.iter().enumerate() {
            if tidx >= t_max { break; }
            let (p,q) = if e8_plane == "jx" {
                let d = b + e;
                let j = b * d;
                let x = e * d;
                (j, x)
            } else {
                (*b, *e)
            };
            v[2*tidx] = p; v[2*tidx+1] = q;
        }
        phi.push(v);
    }
    phi
}

fn modulo(x: f64, m: f64) -> f64 {
    if m == 0.0 { return 0.0; }
    let r = x % m; if r < 0.0 { r + m } else { r }
}

fn zscore_features(phi: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = phi.len(); if n==0 { return vec![]; }
    let d = phi[0].len();
    let mut means = vec![0.0; d];
    for i in 0..n { for j in 0..d { means[j] += phi[i][j]; } }
    for j in 0..d { means[j] /= n as f64; }
    let mut stds = vec![0.0; d];
    for i in 0..n { for j in 0..d { let z = phi[i][j] - means[j]; stds[j] += z*z; } }
    for j in 0..d { stds[j] = (stds[j] / (n.max(1) as f64)).sqrt(); if stds[j] == 0.0 { stds[j] = 1.0; } }
    let mut out = vec![vec![0.0; d]; n];
    for i in 0..n { for j in 0..d { out[i][j] = (phi[i][j] - means[j]) / stds[j]; } }
    out
}

fn apply_phase_mode(phi: &Vec<Vec<f64>>, mode: &str) -> Vec<Vec<f64>> {
    if phi.is_empty() { return phi.clone(); }
    let d = phi[0].len();
    if d < 4 { return phi.clone(); }
    // Last 4 dims are [phase24, phase9, phase72, phase24_bin]
    match mode {
        "none" => phi.iter().map(|row| row[..d-4].to_vec()).collect(),
        "raw" => phi.clone(),
        "sincos" => {
            let mut out: Vec<Vec<f64>> = Vec::with_capacity(phi.len());
            for row in phi {
                let base = &row[..d-4];
                let p24 = row[d-4];
                let p9  = row[d-3];
                let p72 = row[d-2];
                let mut v = Vec::with_capacity(base.len() + 6);
                v.extend_from_slice(base);
                let tpi = std::f64::consts::TAU; // 2π
                v.push((tpi * (p24/24.0)).sin()); v.push((tpi * (p24/24.0)).cos());
                v.push((tpi * (p9 / 9.0)).sin()); v.push((tpi * (p9 / 9.0)).cos());
                v.push((tpi * (p72/72.0)).sin()); v.push((tpi * (p72/72.0)).cos());
                out.push(v);
            }
            out
        }
        _ => phi.clone(),
    }
}

fn select_best_tau(
    phi_scaled: &Vec<Vec<f64>>, mask_adj: &Vec<Vec<f64>>, names: &Vec<String>, k_list: &Vec<usize>,
    gt_raw: &Vec<Option<usize>>, baseline_q: f64, sweep_scales: &Vec<f64>, tau0: f64
) -> (f64, usize, f64, Vec<usize>, Vec<usize>) {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_tau = tau0;
    let mut best_k = 0usize; let mut best_q = f64::NEG_INFINITY; let mut best_sizes = Vec::new(); let mut best_labels = Vec::new();
    let has_labels = gt_raw.iter().all(|x| x.is_some());
    for s in sweep_scales {
        let tau = tau0 * *s;
        let w = build_kernel_adjacency_masked(phi_scaled, tau, mask_adj);
        let (k_s, q_s, sizes_s, labels_s) = try_spectral(&w, k_list);
        let (pur_s, ari_s, nmi_s) = if has_labels { clustering_metrics(&labels_s, &compress_gt(gt_raw).unwrap(), k_s) } else { (0.0,0.0,0.0) };
        let avg_label = if has_labels { (pur_s + ari_s + nmi_s) / 3.0 } else { 0.0 };
        let lambda = if has_labels { 0.2 } else { 0.0 };
        let mut score = q_s + lambda * avg_label;
        if q_s < baseline_q + 0.01 { score -= 1.0; }
        let n = names.len() as f64; let max_share = (sizes_s.iter().cloned().max().unwrap_or(0) as f64)/n;
        if max_share > 0.8 { score -= (max_share - 0.8) * 10.0; }
        if score > best_score { best_score=score; best_tau=tau; best_k=k_s; best_q=q_s; best_sizes=sizes_s; best_labels=labels_s; }
    }
    (best_tau, best_k, best_q, best_sizes, best_labels)
}

fn estimate_tau(phi_z: &Vec<Vec<f64>>) -> f64 {
    // Median of pairwise squared distances (robust)
    let n = phi_z.len(); if n<2 { return 1.0; }
    let d = phi_z[0].len();
    let mut d2s: Vec<f64> = Vec::with_capacity(n*(n-1)/2);
    for i in 0..n { for j in (i+1)..n { let mut s=0.0; for k in 0..d { let t = phi_z[i][k]-phi_z[j][k]; s+=t*t; } d2s.push(s); } }
    d2s.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let mid = d2s.len()/2; let tau = d2s[mid].max(1e-6);
    tau
}

fn build_kernel_adjacency_masked(phi_z: &Vec<Vec<f64>>, tau: f64, mask_adj: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = phi_z.len(); if n==0 { return vec![]; }
    let d = phi_z[0].len();
    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            if mask_adj[i][j] <= 0.0 { continue; }
            let mut s=0.0; for k in 0..d { let t = phi_z[i][k]-phi_z[j][k]; s+=t*t; }
            let val = (-s / tau).exp();
            w[i][j]=val; w[j][i]=val;
        }
    }
    w
}

fn try_spectral(w: &Vec<Vec<f64>>, k_list: &Vec<usize>) -> (usize, f64, Vec<usize>, Vec<usize>) {
    let n = w.len();
    let d: Vec<f64> = (0..n).map(|i| w[i].iter().sum::<f64>()).collect();
    let mut best_q = f64::NEG_INFINITY; let mut best_k=0usize; let mut best_sizes=Vec::new(); let mut best_labels=Vec::new();
    for &k in k_list {
        // M = D^{-1/2} W D^{-1/2}
        let sqrt_inv_d: Vec<f64> = d.iter().map(|&di| if di>0.0 { 1.0/di.sqrt() } else { 0.0 }).collect();
        let m_apply = |v: &Vec<f64>| -> Vec<f64> {
            let n = v.len(); let mut y = vec![0.0; n];
            // y = D^{-1/2} W D^{-1/2} v
            let mut tmp = vec![0.0; n];
            for j in 0..n { tmp[j] = sqrt_inv_d[j]*v[j]; }
            for i in 0..n {
                let mut s = 0.0; let wi = &w[i];
                for j in 0..n { let wij = wi[j]; if wij != 0.0 { s += wij * tmp[j]; } }
                y[i] = sqrt_inv_d[i]*s;
            }
            y
        };
        let u = top_k_eigenvectors(&m_apply, n, k, 100, 1); // 1 = deterministic seed id
        let embed = transpose(&u); // n x k
        let labels = kmeans(&embed, k, 20, 5); // 20 iters, 5 restarts
        let q = modularity(w, &d, &labels);
        if q > best_q { best_q = q; best_k = k; best_labels = labels; best_sizes = cluster_sizes(&best_labels, k); }
    }
    (best_k, best_q, best_sizes, best_labels)
}

fn cluster_sizes(labels: &Vec<usize>, k: usize) -> Vec<usize> { let mut c=vec![0usize; k]; for &l in labels { if l<k { c[l]+=1; } } c }

fn modularity(w: &Vec<Vec<f64>>, d: &Vec<f64>, labels: &Vec<usize>) -> f64 {
    let n = w.len();
    let m2: f64 = d.iter().sum::<f64>(); // 2m for undirected
    if m2 <= 0.0 { return 0.0; }
    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            if labels[i] == labels[j] {
                q += w[i][j] - (d[i]*d[j]/m2);
            }
        }
    }
    q / m2
}

fn top_k_eigenvectors<F>(m_apply: &F, n: usize, k: usize, iters: usize, seed: u64) -> Vec<Vec<f64>> where F: Fn(&Vec<f64>)->Vec<f64> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut rng = XorShift64::new(seed);
    for _ in 0..k {
        let mut v = vec![0.0; n];
        for i in 0..n { v[i] = (rng.next_f64() - 0.5); }
        normalize(&mut v);
        for _ in 0..iters {
            let mut y = m_apply(&v);
            // orthonormalize vs existing basis
            for u in &basis { let proj = dot(&y, u); axpy(&mut y, u, -proj); }
            normalize(&mut y);
            v = y;
        }
        // final orthonormalization
        for u in &basis { let proj = dot(&v, u); axpy(&mut v, u, -proj); }
        normalize(&mut v);
        basis.push(v);
    }
    basis // k vectors of length n (columns)
}

fn kmeans(points: &Vec<Vec<f64>>, k: usize, iters: usize, restarts: usize) -> Vec<usize> {
    let n = points.len(); let d = points[0].len();
    let mut best_labels = vec![0usize; n]; let mut best_inertia = f64::INFINITY;
    let mut rng = XorShift64::new(42);
    for _ in 0..restarts {
        let mut centers = kmeans_pp_init(points, k, &mut rng);
        let mut labels = vec![0usize; n];
        for _ in 0..iters {
            // assign
            for i in 0..n { labels[i] = argmin_center(&points[i], &centers); }
            // update
            let mut sums = vec![vec![0.0; d]; k];
            let mut counts = vec![0usize; k];
            for i in 0..n { let c = labels[i]; counts[c]+=1; for j in 0..d { sums[c][j]+=points[i][j]; } }
            for c in 0..k { if counts[c]>0 { for j in 0..d { sums[c][j]/=counts[c] as f64; } centers[c]=sums[c].clone(); } }
        }
        let inertia = total_inertia(points, &labels, &centers);
        if inertia < best_inertia { best_inertia = inertia; best_labels = labels; }
    }
    best_labels
}

fn kmeans_pp_init(points: &Vec<Vec<f64>>, k: usize, rng: &mut XorShift64) -> Vec<Vec<f64>> {
    let n = points.len(); let d = points[0].len();
    let mut centers: Vec<Vec<f64>> = Vec::new();
    centers.push(points[0].clone()); // deterministic first center
    let mut dist2: Vec<f64> = vec![f64::INFINITY; n];
    for _ in 1..k {
        // update distances to nearest center
        for i in 0..n { let mut best = f64::INFINITY; for c in &centers { let ds = squared_dist(&points[i], c); if ds<best { best=ds; } } dist2[i]=best; }
        let sumd: f64 = dist2.iter().sum();
        let mut r = rng.next_f64() * sumd;
        let mut idx = 0usize; let mut acc=0.0;
        while idx<n-1 && acc + dist2[idx] < r { acc += dist2[idx]; idx += 1; }
        centers.push(points[idx].clone());
    }
    centers
}

fn argmin_center(p: &Vec<f64>, centers: &Vec<Vec<f64>>) -> usize { let mut best=0usize; let mut bestd=f64::INFINITY; for (i,c) in centers.iter().enumerate() { let d=squared_dist(p,c); if d<bestd { bestd=d; best=i; } } best }
fn squared_dist(a: &Vec<f64>, b: &Vec<f64>) -> f64 { let mut s=0.0; for i in 0..a.len() { let d=a[i]-b[i]; s+=d*d; } s }
fn total_inertia(points: &Vec<Vec<f64>>, labels: &Vec<usize>, centers: &Vec<Vec<f64>>) -> f64 { let mut s=0.0; for i in 0..points.len() { s+= squared_dist(&points[i], &centers[labels[i]]); } s }

fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 { let mut s=0.0; for i in 0..a.len() { s+=a[i]*b[i]; } s }
fn axpy(y: &mut Vec<f64>, x: &Vec<f64>, a: f64) { for i in 0..y.len() { y[i] += a*x[i]; } }
fn normalize(v: &mut Vec<f64>) { let mut s=0.0; for &x in v.iter() { s+=x*x; } let n=s.sqrt(); if n>0.0 { for i in 0..v.len() { v[i]/=n; } } }
fn transpose(cols: &Vec<Vec<f64>>) -> Vec<Vec<f64>> { if cols.is_empty() { return vec![]; } let n=cols[0].len(); let k=cols.len(); let mut m=vec![vec![0.0;k]; n]; for (j, col) in cols.iter().enumerate() { for i in 0..n { m[i][j]=col[i]; } } m }

fn json_list_usize(v: &Vec<usize>) -> String { let mut s=String::new(); s.push('['); for (i,x) in v.iter().enumerate() { if i>0 { s.push_str(", "); } s.push_str(&x.to_string()); } s.push(']'); s }
fn json_list_prefix(v: &Vec<usize>, n: usize) -> String { let mut s=String::new(); s.push('['); let mut first=true; for (i,x) in v.iter().enumerate() { if i>=n { break; } if !first { s.push_str(", "); } first=false; s.push_str(&x.to_string()); } s.push(']'); s }
fn escape<S: AsRef<str>>(s: S) -> String { let mut out=String::new(); for c in s.as_ref().chars() { match c { '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"), x if x.is_control() => out.push(' '), x => out.push(x), } } out }
fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Baseline => "baseline",
        Mode::X => "qa_weight_x",
        Mode::J => "qa_weight_j",
        Mode::Mix(_) => "qa_weight_mix",
        Mode::Full => "qa_weight_full",
        Mode::FullMulti => "qa_weight_full_multi",
        Mode::E8Embed => "qa_weight_e8_embed",
    }
}

struct XorShift64 { state: u64 }
impl XorShift64 { fn new(seed: u64) -> Self { let s = if seed==0 { 88172645463393265 } else { seed }; Self { state: s } } fn next_u64(&mut self) -> u64 { let mut x=self.state; x^=x<<13; x^=x>>7; x^=x<<17; self.state=x; x } fn next_f64(&mut self) -> f64 { (self.next_u64() as f64) / (u64::MAX as f64) } }

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qa_features_canonical_1123() {
        // (b,e,d,a) = (1,1,2,3)
        let f = build_canonical_features(1.0, 1.0, 2.0, 3.0);
        // C = 2ed = 4, F = ba = 3, G = e^2 + d^2 = 1 + 4 = 5
        assert!(approx_eq(f.c, 4.0, 1e-9));
        assert!(approx_eq(f.f, 3.0, 1e-9));
        assert!(approx_eq(f.g, 5.0, 1e-9));
        // J=bd=2, X=ed=2, K=da=6, W=X+K=8, Y=A-D=9-4=5, Z=E+K=1+6=7
        assert!(approx_eq(f.j, 2.0, 1e-9));
        assert!(approx_eq(f.x, 2.0, 1e-9));
        assert!(approx_eq(f.k, 6.0, 1e-9));
        assert!(approx_eq(f.w, 8.0, 1e-9));
        assert!(approx_eq(f.y, 5.0, 1e-9));
        assert!(approx_eq(f.z, 7.0, 1e-9));
        // Pythagorean closure
        assert!(approx_eq(f.c*f.c + f.f*f.f, f.g*f.g, 1e-9));
    }

    #[test]
    fn qa_features_canonical_1235() {
        // (b,e,d,a) = (1,2,3,5)
        let f = build_canonical_features(1.0, 2.0, 3.0, 5.0);
        // C=2ed=12, F=ba=5, G=e^2+d^2=4+9=13
        assert!(approx_eq(f.c, 12.0, 1e-9));
        assert!(approx_eq(f.f, 5.0, 1e-9));
        assert!(approx_eq(f.g, 13.0, 1e-9));
        // J=bd=3, X=ed=6, K=da=15, W=X+K=21, Y=A-D=25-9=16, Z=E+K=4+15=19
        assert!(approx_eq(f.j, 3.0, 1e-9));
        assert!(approx_eq(f.x, 6.0, 1e-9));
        assert!(approx_eq(f.k, 15.0, 1e-9));
        assert!(approx_eq(f.w, 21.0, 1e-9));
        assert!(approx_eq(f.y, 16.0, 1e-9));
        assert!(approx_eq(f.z, 19.0, 1e-9));
        assert!(approx_eq(f.c*f.c + f.f*f.f, f.g*f.g, 1e-9));
    }
}

fn write_labels_csv(path: &PathBuf, names: &Vec<String>, labels: &Vec<usize>, deg: &Vec<f64>, cc: &Vec<f64>, harm: &Vec<f64>, jv: &Vec<f64>, xv: &Vec<f64>) -> io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "node,label,degree,cc,harm,J,X")?;
    for i in 0..names.len() {
        writeln!(f, "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}", escape_csv(&names[i]), labels[i], deg[i], cc[i], harm[i], jv[i], xv[i])?;
    }
    Ok(())
}

fn write_cluster_stats_json(path: &PathBuf, k: usize, labels: &Vec<usize>, deg: &Vec<f64>, cc: &Vec<f64>, harm: &Vec<f64>, jv: &Vec<f64>, xv: &Vec<f64>) -> io::Result<()> {
    let mut counts = vec![0usize; k];
    let mut sum_deg = vec![0.0; k];
    let mut sum_cc = vec![0.0; k];
    let mut sum_harm = vec![0.0; k];
    let mut sum_j = vec![0.0; k];
    let mut sum_x = vec![0.0; k];
    for i in 0..labels.len() {
        let c = labels[i]; if c>=k { continue; }
        counts[c]+=1; sum_deg[c]+=deg[i]; sum_cc[c]+=cc[i]; sum_harm[c]+=harm[i]; sum_j[c]+=jv[i]; sum_x[c]+=xv[i];
    }
    let mut json = String::new();
    json.push_str("{\n  \"k\": "); json.push_str(&k.to_string()); json.push_str(",\n  \"clusters\": [\n");
    for c in 0..k {
        let sz = counts[c] as f64;
        let (mdeg, mcc, mharm, mj, mx) = if sz>0.0 {
            (sum_deg[c]/sz, sum_cc[c]/sz, sum_harm[c]/sz, sum_j[c]/sz, sum_x[c]/sz)
        } else { (0.0,0.0,0.0,0.0,0.0) };
        json.push_str(&format!(
            "    {{ \"cluster\": {}, \"size\": {}, \"mean_degree\": {:.6}, \"mean_cc\": {:.6}, \"mean_harm\": {:.6}, \"mean_J\": {:.6}, \"mean_X\": {:.6} }}{}\n",
            c, counts[c], mdeg, mcc, mharm, mj, mx, if c+1==k { "" } else { "," }
        ));
    }
    json.push_str("  ]\n}\n");
    let mut f = File::create(path)?; f.write_all(json.as_bytes())?; Ok(())
}

fn escape_csv<S: AsRef<str>>(s: S) -> String {
    let x = s.as_ref();
    if x.contains(',') || x.contains('"') { format!("\"{}\"", x.replace('"', "\"\"")) } else { x.to_string() }
}

// --- Metrics helpers ---
fn compress_gt(gt_opt: &Vec<Option<usize>>) -> Option<Vec<usize>> {
    if gt_opt.is_empty() || gt_opt.iter().all(|x| x.is_none()) { return None; }
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    let mut out: Vec<usize> = Vec::with_capacity(gt_opt.len());
    for v in gt_opt.iter() {
        match v {
            Some(val) => {
                let idx = *map.entry(*val).or_insert_with(|| { let t=next; next+=1; t });
                out.push(idx);
            }
            None => { return None; }
        }
    }
    Some(out)
}

fn clustering_metrics(pred: &Vec<usize>, gt: &Vec<usize>, k: usize) -> (f64, f64, f64) {
    let n = pred.len();
    let c = 1 + gt.iter().max().cloned().unwrap_or(0);
    let mut cm = vec![vec![0usize; c]; k];
    for i in 0..n { let p = pred[i]; let g = gt[i]; if p<k && g<c { cm[p][g]+=1; } }
    let purity = purity_from_cm(&cm, n);
    let ari = ari_from_cm(&cm, n);
    let nmi = nmi_from_cm(&cm, n);
    (purity, ari, nmi)
}

fn comb2(x: usize) -> f64 { if x<2 { 0.0 } else { (x as f64)*(x as f64 - 1.0)/2.0 } }

fn purity_from_cm(cm: &Vec<Vec<usize>>, n: usize) -> f64 {
    let mut sum_max = 0usize;
    for row in cm {
        let mut m = 0usize; for &v in row { if v>m { m=v; } }
        sum_max += m;
    }
    sum_max as f64 / n as f64
}

fn ari_from_cm(cm: &Vec<Vec<usize>>, n: usize) -> f64 {
    let mut sum_comb = 0.0;
    let mut a_sum = 0.0; // sum over clusters comb2(a_i)
    let mut b_sum = 0.0; // sum over classes comb2(b_j)
    let k = cm.len(); let c = if k>0 { cm[0].len() } else { 0 };
    let mut col_sums = vec![0usize; c];
    for i in 0..k {
        let mut row_sum = 0usize;
        for j in 0..c { let nij = cm[i][j]; sum_comb += comb2(nij); row_sum += nij; col_sums[j]+=nij; }
        a_sum += comb2(row_sum);
    }
    for j in 0..c { b_sum += comb2(col_sums[j]); }
    let total = comb2(n);
    if total == 0.0 { return 0.0; }
    let expected = (a_sum * b_sum) / total;
    let max_idx = 0.5 * (a_sum + b_sum);
    let num = sum_comb - expected;
    let den = max_idx - expected;
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

fn nmi_from_cm(cm: &Vec<Vec<usize>>, n: usize) -> f64 {
    let k = cm.len(); let c = if k>0 { cm[0].len() } else { 0 };
    if n == 0 || k == 0 || c == 0 { return 0.0; }
    let mut row_sums = vec![0usize; k];
    let mut col_sums = vec![0usize; c];
    for i in 0..k { for j in 0..c { row_sums[i]+=cm[i][j]; col_sums[j]+=cm[i][j]; } }
    let nf = n as f64;
    let mut mi = 0.0;
    for i in 0..k {
        for j in 0..c {
            let nij = cm[i][j]; if nij == 0 { continue; }
            let p_ij = (nij as f64)/nf; let p_i = (row_sums[i] as f64)/nf; let p_j = (col_sums[j] as f64)/nf;
            mi += p_ij * (p_ij/(p_i*p_j)).ln();
        }
    }
    let h_u = - row_sums.iter().filter(|&&x| x>0).map(|&x| { let p=(x as f64)/nf; p * p.ln() }).sum::<f64>();
    let h_v = - col_sums.iter().filter(|&&x| x>0).map(|&x| { let p=(x as f64)/nf; p * p.ln() }).sum::<f64>();
    let denom = (h_u * h_v).sqrt();
    if denom <= 0.0 { 0.0 } else { mi / denom }
}

fn opt_f64_json(v: Option<f64>) -> String { match v { Some(x) => format!("{:.6}", x), None => "null".to_string() } }
