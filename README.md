# Author: GHawk1124 (Garrett Comes) 4/23/2024
# For AE 4132: Finite Element Analysis Final Project

# FEM Simulation

Try it online: [https://GHawk1124.github.io/fem_rust_ae](https://GHawk1124.github.io/fem_rust_ae)

## Dependencies

### Windows
- Rust toolchain (install from [rustup.rs](https://rustup.rs))
- Visual Studio Build Tools with C++ workload

### Linux
- Rust toolchain (install from [rustup.rs](https://rustup.rs))
- Build essentials: `sudo apt install build-essential`
- Required libraries: `sudo apt install libx11-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev`

## Building

### Native (Windows/Linux)
```bash
cargo build --release
```

### Web (WASM)
```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen CLI
cargo install wasm-bindgen-cli

# Build WASM binary
cargo build --release --target wasm32-unknown-unknown

# Generate JS bindings (run from project root)
wasm-bindgen --out-dir ./out/ --target web ./target/wasm32-unknown-unknown/release/fem_sim.wasm
```
The generated files in `./out/` directory can be served using any web server.

## Convergence Analysis
To run the convergence study (not available in WASM):
```bash
cargo run --release -- convergence
```
This will generate convergence plots in `./output/convergence_study.png` and `./output/convergence_study.svg`.
