fn main() {
    tonic_build::configure()
        .build_client(false)
        .compile(&["proto/service.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
