//! QA-AlphaGeometry CLI
//!
//! TODO: Implement CLI after core is complete

use clap::Parser;

#[derive(Parser)]
#[command(name = "qa-geo-solve")]
#[command(about = "Solve geometry problems using QA priors")]
struct Args {
    /// Input problem file (JSON)
    problem: String,
}

fn main() {
    let _args = Args::parse();
    println!("QA-AlphaGeometry solver (not yet implemented)");
}
