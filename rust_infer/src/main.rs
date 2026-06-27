// Minimal Rust seal-inference stub using ONNX Runtime (`ort`).
// Proves the deployment path: load seal_lite[.|_int8].onnx, run, benchmark CPU.
//   cargo run --release -- <model.onnx> [iters]
use std::time::Instant;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

fn main() -> ort::Result<()> {
    let model = std::env::args().nth(1).expect("usage: seal_infer <model.onnx> [iters]");
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(&model)?;

    let (n, c, h, w) = (1usize, 3usize, 384usize, 384usize);
    let make = || Tensor::from_array(([n, c, h, w], vec![0.1f32; n * c * h * w]));

    // warmup
    for _ in 0..5 {
        let _ = session.run(ort::inputs!["input" => make()?])?;
    }
    // timed
    let t = Instant::now();
    let mut shape = String::new();
    for _ in 0..iters {
        let outputs = session.run(ort::inputs!["input" => make()?])?;
        let (sh, _data) = outputs["logits"].try_extract_tensor::<f32>()?;
        shape = format!("{:?}", sh);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("model={model}  threads={threads}  iters={iters}");
    println!("-> {ms:.1} ms/inference   out_shape={shape}");
    Ok(())
}
