fn main() {
    if let Err(error) = drysua::run_from_env() {
        eprintln!("drysua: {error}");
        std::process::exit(1);
    }
}
