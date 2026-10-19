//! Links a system LAPACK only when the `lapack` feature is on (see Cargo.toml); otherwise does nothing.
fn main() {
    println!("cargo:rerun-if-env-changed=PSFZ_LAPACK_LIB");
    println!("cargo:rerun-if-env-changed=PSFZ_LAPACK_DIR");
    if std::env::var_os("CARGO_FEATURE_LAPACK").is_none() {
        return;
    }
    if let Ok(dir) = std::env::var("PSFZ_LAPACK_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    let lib = std::env::var("PSFZ_LAPACK_LIB").unwrap_or_else(|_| "liblapack.so.3".into());
    println!("cargo:rustc-link-lib=dylib:+verbatim={}", lib);
}
