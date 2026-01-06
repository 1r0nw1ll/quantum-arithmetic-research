// QARM v0.2 Constitutional Verification Runner
// Executes GLFSI theorem verification

use qarm_constitutional::verify_glfsi;

fn main() {
    println!("QARM v0.2 Constitutional Verification");
    println!("Rust Mirror of QARM_v02_Failures.tla");
    println!();

    match verify_glfsi() {
        Ok(()) => {
            println!();
            println!("✅ Constitutional verification PASSED");
            println!("   QARM v0.2 Rust implementation matches TLA+ spec exactly.");
            std::process::exit(0);
        }
        Err(e) => {
            println!();
            println!("❌ Constitutional verification FAILED");
            println!("   Error: {}", e);
            println!("   STOP: Constitution violated - do not proceed to production.");
            std::process::exit(1);
        }
    }
}
