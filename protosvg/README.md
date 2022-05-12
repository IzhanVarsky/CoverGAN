# ProtoSVG

ProtoSVG is a subset of the SVG format presented as protocol buffers. This repository includes a format description and
a gRPC service capable of converting ProtoSVG to SVG and rendering it.

## Dependencies

* `resvg` and its component libraries: SVG creation and rendering
* `prost`: ProtoBuf code generation
* `tonic`: gRPC server implementation

## Building and running

This library relies on the standard Rust toolchain and commands.

* Debug: `cargo run`
* Release: `cargo run --release`
* Installation to PATH: `cargo install --locked --path .`
* Linting: `cargo clippy`

## Usage

The `client/` subdirectory includes a demo Python client.  
With the `grpc-tools` Python package installed, execute `./generate_pb.sh` to generate Python code for the format.  
After that, running `./client.py` should display a blue square and print its corresponding SVG definition.
