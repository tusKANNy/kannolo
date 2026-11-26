fn main() {
    // Mark a build made with the `dev-python` profile (more codegen units, slower search). The
    // Python package uses the mark to warn about it. Cargo names the profile only in the output
    // path.
    println!("cargo::rustc-check-cfg=cfg(kannolo_dev_profile)");
    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    if std::path::Path::new(&out_dir)
        .components()
        .any(|c| c.as_os_str() == "dev-python")
    {
        println!("cargo:rustc-cfg=kannolo_dev_profile");
    }

    // On macOS, Python symbols are resolved at runtime when the extension is loaded.
    // The Apple linker requires explicit permission to leave them undefined.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
