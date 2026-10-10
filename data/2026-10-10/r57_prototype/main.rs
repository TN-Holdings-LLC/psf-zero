//! Cross-check driver (prototype only): reads buffers from files, prints each result.
mod statevec;
use statevec::*;

fn main() {
    for path in std::env::args().skip(1) {
        let buf = std::fs::read(&path).expect("read");
        match parse(&buf) {
            Err(e) => println!("{} ERROR {}", path, e),
            Ok((Kind::Excitation, p, _)) => println!("{} {:.17e}", path, excitation_cost(&p)),
            Ok((Kind::HybridGates, p, _)) => println!("{} {:.17e}", path, hybrid_cost_gates(&p)),
            Ok((Kind::ApplyOps, p, Some(mut s))) => {
                apply_ops(&mut s, p.k, &p.ops);
                std::fs::write(format!("{}.out", path), state_bytes(&s)).expect("write");
                println!("{} state", path);
            }
            Ok(_) => println!("{} ERROR no state", path),
        }
    }
}
