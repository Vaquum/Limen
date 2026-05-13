use std::env;
use std::fs;
use std::process;
use std::time::Instant;

use ridge_core::{alpha_for, RidgeDataset, RidgeModel, RidgeProblem};

fn main() {
    if let Err(err) = run() {
        eprintln!("ridge_core: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return Err("usage: ridge_core <inspect|bench> <data.bin> [alpha|counts]".into());
    }

    let command = &args[1];
    let dataset = load_dataset(&args[2])?;

    match command.as_str() {
        "inspect" => {
            let alpha = args
                .get(3)
                .map(|value| value.parse::<f64>())
                .transpose()?
                .unwrap_or(1.0);
            let problem = dataset.problem()?;
            let model = problem.train(alpha)?;
            print_inspect(&problem, &model);
        }
        "bench" => {
            let counts = args
                .get(3)
                .map(|value| parse_counts(value))
                .transpose()?
                .unwrap_or_else(|| vec![10, 100, 1_000, 10_000, 100_000]);
            print_dataset(&dataset);
            for count in counts {
                run_block(&dataset, count)?;
            }
        }
        _ => return Err(format!("unknown command {command:?}").into()),
    }

    Ok(())
}

fn load_dataset(path: &str) -> Result<RidgeDataset, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(RidgeDataset::from_binary(&bytes)?)
}

fn parse_counts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for item in value.split(',') {
        out.push(item.trim().parse::<usize>()?);
    }
    Ok(out)
}

fn run_block(dataset: &RidgeDataset, count: usize) -> Result<(), Box<dyn std::error::Error>> {
    if count == 0 {
        return Err("count must be positive".into());
    }

    let start = Instant::now();
    let mut checksum = 0.0;
    let mut last_model = None;
    let mut alpha_min = f64::INFINITY;
    let mut alpha_max = f64::NEG_INFINITY;

    for i in 0..count {
        let alpha = alpha_for(i);
        alpha_min = alpha_min.min(alpha);
        alpha_max = alpha_max.max(alpha);
        let problem = dataset.problem()?;
        let model = problem.train(alpha)?;
        checksum += model.r2 + model.coeff_norm + model.beta[0] * 0.000_001;
        last_model = Some(model);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let model = last_model.unwrap();
    println!(
        "{{\"event\":\"bench\",\"mode\":\"full_training_rebuild_stats\",\"count\":{},\"seconds\":{:.9},\"us_per_iter\":{:.6},\"alpha_min\":{:.12},\"alpha_max\":{:.12},\"last_alpha\":{:.12},\"last_mse\":{:.12},\"last_r2\":{:.12},\"last_coeff_norm\":{:.12},\"checksum\":{:.12}}}",
        count,
        elapsed,
        elapsed * 1_000_000.0 / count as f64,
        alpha_min,
        alpha_max,
        model.alpha,
        model.mse,
        model.r2,
        model.coeff_norm,
        checksum
    );
    Ok(())
}

fn print_dataset(dataset: &RidgeDataset) {
    let ws = dataset.working_set();
    println!(
        "{{\"event\":\"problem\",\"rows\":{},\"cols\":{},\"input_matrix_bytes\":{},\"precomputed_stats_bytes\":{},\"solver_scratch_bytes\":{},\"full_training_live_bytes\":{}}}",
        dataset.rows(),
        dataset.cols(),
        ws.input_matrix_bytes,
        ws.precomputed_stats_bytes,
        ws.solver_scratch_bytes,
        ws.full_training_live_bytes
    );
}

fn print_problem(problem: &RidgeProblem) {
    let ws = problem.working_set();
    println!(
        "{{\"event\":\"problem\",\"rows\":{},\"cols\":{},\"input_matrix_bytes\":{},\"precomputed_stats_bytes\":{},\"solver_scratch_bytes\":{},\"full_training_live_bytes\":{}}}",
        problem.rows(),
        problem.cols(),
        ws.input_matrix_bytes,
        ws.precomputed_stats_bytes,
        ws.solver_scratch_bytes,
        ws.full_training_live_bytes
    );
}

fn print_inspect(problem: &RidgeProblem, model: &RidgeModel) {
    print_problem(problem);
    print!(
        "{{\"event\":\"inspect\",\"alpha\":{:.12},\"mse\":{:.12},\"r2\":{:.12},\"coeff_norm\":{:.12},\"beta\":[",
        model.alpha, model.mse, model.r2, model.coeff_norm
    );
    for (idx, value) in model.beta.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!("{:.17}", value);
    }
    println!("]}}");
}
