use std::convert::{TryFrom, TryInto};
use std::fmt;

const MAGIC: &[u8; 8] = b"RIDGE001";
const F64_BYTES: usize = 8;

#[derive(Debug)]
pub enum RidgeError {
    InvalidInput(String),
    SingularMatrix,
}

impl fmt::Display for RidgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RidgeError::InvalidInput(msg) => write!(f, "{msg}"),
            RidgeError::SingularMatrix => write!(f, "ridge system is singular"),
        }
    }
}

impl std::error::Error for RidgeError {}

#[derive(Debug, Clone)]
pub struct RidgeProblem {
    rows: usize,
    cols: usize,
    gram: Vec<f64>,
    rhs: Vec<f64>,
    y_sum: f64,
    y_sq_sum: f64,
}

#[derive(Debug, Clone)]
pub struct RidgeModel {
    pub alpha: f64,
    pub beta: Vec<f64>,
    pub mse: f64,
    pub r2: f64,
    pub coeff_norm: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkingSet {
    pub input_matrix_bytes: usize,
    pub precomputed_stats_bytes: usize,
    pub solver_scratch_bytes: usize,
    pub full_training_live_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct RidgeDataset {
    rows: usize,
    cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl RidgeDataset {
    pub fn from_row_major(
        rows: usize,
        cols: usize,
        x: &[f64],
        y: &[f64],
    ) -> Result<Self, RidgeError> {
        validate_dimensions(rows, cols, x.len(), y.len())?;
        Ok(Self {
            rows,
            cols,
            x: x.to_vec(),
            y: y.to_vec(),
        })
    }

    pub fn from_binary(bytes: &[u8]) -> Result<Self, RidgeError> {
        if bytes.len() < 24 {
            return Err(RidgeError::InvalidInput("file is too short".to_string()));
        }
        if &bytes[0..8] != MAGIC {
            return Err(RidgeError::InvalidInput(
                "invalid ridge data magic".to_string(),
            ));
        }

        let rows = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap()))
            .map_err(|_| RidgeError::InvalidInput("rows do not fit usize".to_string()))?;
        let cols = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
            .map_err(|_| RidgeError::InvalidInput("cols do not fit usize".to_string()))?;
        let x_count = checked_mul(rows, cols, "rows*cols overflow")?;
        validate_dimensions(rows, cols, x_count, rows)?;

        let value_count = checked_add(x_count, rows, "value count overflow")?;
        let payload_bytes = checked_mul(value_count, F64_BYTES, "payload size overflow")?;
        let expected = checked_add(24, payload_bytes, "file size overflow")?;
        if bytes.len() != expected {
            return Err(RidgeError::InvalidInput(format!(
                "file size {} does not match expected {}",
                bytes.len(),
                expected
            )));
        }

        let mut offset = 24;
        let mut x = Vec::with_capacity(x_count);
        for _ in 0..x_count {
            x.push(read_f64(bytes, &mut offset));
        }
        let mut y = Vec::with_capacity(rows);
        for _ in 0..rows {
            y.push(read_f64(bytes, &mut offset));
        }
        Ok(Self { rows, cols, x, y })
    }

    pub fn problem(&self) -> Result<RidgeProblem, RidgeError> {
        RidgeProblem::from_row_major(self.rows, self.cols, &self.x, &self.y)
    }

    pub fn working_set(&self) -> WorkingSet {
        compute_working_set(self.rows, self.cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl RidgeProblem {
    pub fn from_row_major(
        rows: usize,
        cols: usize,
        x: &[f64],
        y: &[f64],
    ) -> Result<Self, RidgeError> {
        validate_dimensions(rows, cols, x.len(), y.len())?;

        let dim = checked_dim(cols)?;
        let matrix_len = checked_mul(dim, dim, "normal equation matrix overflow")?;
        let mut gram = vec![0.0; matrix_len];
        let mut rhs = vec![0.0; dim];
        let mut y_sum = 0.0;
        let mut y_sq_sum = 0.0;

        gram[0] = rows as f64;
        for row in 0..rows {
            let yv = y[row];
            y_sum += yv;
            y_sq_sum += yv * yv;
            rhs[0] += yv;

            let row_offset = row * cols;
            for col in 0..cols {
                let xv = x[row_offset + col];
                let idx = col + 1;
                gram[idx] += xv;
                gram[idx * dim] += xv;
                rhs[idx] += xv * yv;
            }

            for left in 0..cols {
                let lv = x[row_offset + left];
                let li = left + 1;
                for right in left..cols {
                    let rv = x[row_offset + right];
                    let ri = right + 1;
                    gram[li * dim + ri] += lv * rv;
                }
            }
        }

        for left in 1..dim {
            for right in (left + 1)..dim {
                gram[right * dim + left] = gram[left * dim + right];
            }
        }

        Ok(Self {
            rows,
            cols,
            gram,
            rhs,
            y_sum,
            y_sq_sum,
        })
    }

    pub fn from_binary(bytes: &[u8]) -> Result<Self, RidgeError> {
        RidgeDataset::from_binary(bytes)?.problem()
    }

    pub fn train(&self, alpha: f64) -> Result<RidgeModel, RidgeError> {
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(RidgeError::InvalidInput(format!("invalid alpha {alpha}")));
        }

        let dim = self.cols + 1;
        let mut system = self.gram.clone();
        for idx in 1..dim {
            system[idx * dim + idx] += alpha;
        }

        let beta = solve_linear(system, self.rhs.clone(), dim)?;
        let mse = self.sse(&beta).max(0.0) / self.rows as f64;
        let tss = (self.y_sq_sum - (self.y_sum * self.y_sum / self.rows as f64)).max(0.0);
        let r2 = if tss > 0.0 {
            1.0 - (mse * self.rows as f64 / tss)
        } else {
            0.0
        };
        let coeff_norm = beta[1..].iter().map(|v| v * v).sum::<f64>().sqrt();

        Ok(RidgeModel {
            alpha,
            beta,
            mse,
            r2,
            coeff_norm,
        })
    }

    pub fn working_set(&self) -> WorkingSet {
        compute_working_set(self.rows, self.cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn sse(&self, beta: &[f64]) -> f64 {
        let dim = self.cols + 1;
        let mut beta_rhs = 0.0;
        for i in 0..dim {
            beta_rhs += beta[i] * self.rhs[i];
        }

        let mut beta_gram_beta = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                beta_gram_beta += beta[i] * self.gram[i * dim + j] * beta[j];
            }
        }

        self.y_sq_sum - 2.0 * beta_rhs + beta_gram_beta
    }
}

pub fn write_binary(rows: usize, cols: usize, x: &[f64], y: &[f64]) -> Result<Vec<u8>, RidgeError> {
    validate_dimensions(rows, cols, x.len(), y.len())?;
    let value_count = checked_add(x.len(), y.len(), "value count overflow")?;
    let payload_bytes = checked_mul(value_count, F64_BYTES, "payload size overflow")?;
    let total_bytes = checked_add(24, payload_bytes, "file size overflow")?;

    let mut bytes = Vec::with_capacity(total_bytes);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(rows as u64).to_le_bytes());
    bytes.extend_from_slice(&(cols as u64).to_le_bytes());
    for value in x {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in y {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub fn alpha_for(iteration: usize) -> f64 {
    let hashed = iteration.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 100_000;
    let t = hashed as f64 / 99_999.0;
    10.0_f64.powf(-6.0 + 8.0 * t)
}

fn validate_dimensions(
    rows: usize,
    cols: usize,
    x_len: usize,
    y_len: usize,
) -> Result<(), RidgeError> {
    if rows == 0 {
        return Err(RidgeError::InvalidInput(
            "rows must be positive".to_string(),
        ));
    }
    if cols == 0 {
        return Err(RidgeError::InvalidInput(
            "cols must be positive".to_string(),
        ));
    }

    let x_count = checked_mul(rows, cols, "rows*cols overflow")?;
    if x_len != x_count {
        return Err(RidgeError::InvalidInput(format!(
            "x length {x_len} does not match rows*cols {x_count}"
        )));
    }
    if y_len != rows {
        return Err(RidgeError::InvalidInput(format!(
            "y length {y_len} does not match rows {rows}"
        )));
    }

    let dim = checked_dim(cols)?;
    checked_mul(dim, dim, "normal equation matrix overflow")?;
    Ok(())
}

fn compute_working_set(rows: usize, cols: usize) -> WorkingSet {
    let dim = cols + 1;
    let input_matrix_bytes = (rows * cols + rows) * F64_BYTES;
    let precomputed_stats_bytes = (dim * dim + dim) * F64_BYTES;
    let solver_scratch_bytes = (dim * dim + dim) * F64_BYTES;
    let full_training_live_bytes =
        input_matrix_bytes + precomputed_stats_bytes + solver_scratch_bytes;
    WorkingSet {
        input_matrix_bytes,
        precomputed_stats_bytes,
        solver_scratch_bytes,
        full_training_live_bytes,
    }
}

fn checked_dim(cols: usize) -> Result<usize, RidgeError> {
    checked_add(cols, 1, "column count overflow")
}

fn checked_add(left: usize, right: usize, message: &str) -> Result<usize, RidgeError> {
    left.checked_add(right)
        .ok_or_else(|| RidgeError::InvalidInput(message.to_string()))
}

fn checked_mul(left: usize, right: usize, message: &str) -> Result<usize, RidgeError> {
    left.checked_mul(right)
        .ok_or_else(|| RidgeError::InvalidInput(message.to_string()))
}

fn read_f64(bytes: &[u8], offset: &mut usize) -> f64 {
    let value = f64::from_le_bytes(bytes[*offset..*offset + F64_BYTES].try_into().unwrap());
    *offset += F64_BYTES;
    value
}

fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>, RidgeError> {
    let eps = 1e-12;
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = a[row * n + col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }

        if pivot_abs < eps {
            return Err(RidgeError::SingularMatrix);
        }

        if pivot != col {
            for k in col..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }

        let diag = a[col * n + col];
        for k in col..n {
            a[col * n + k] /= diag;
        }
        b[col] /= diag;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_simple_line() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let problem = RidgeProblem::from_row_major(4, 1, &x, &y).unwrap();
        let model = problem.train(1e-9).unwrap();
        assert!((model.beta[0] - 1.0).abs() < 1e-6);
        assert!((model.beta[1] - 2.0).abs() < 1e-6);
        assert!(model.r2 > 0.999_999);
    }

    #[test]
    fn binary_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![1.0, 2.0, 3.0];
        let bytes = write_binary(3, 2, &x, &y).unwrap();
        let problem = RidgeProblem::from_binary(&bytes).unwrap();
        assert_eq!(problem.rows(), 3);
        assert_eq!(problem.cols(), 2);
    }

    #[test]
    fn rejects_overflowing_binary_dimensions() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        let err = RidgeDataset::from_binary(&bytes).unwrap_err();
        assert!(matches!(err, RidgeError::InvalidInput(_)));
    }

    #[test]
    fn alpha_schedule_is_positive() {
        for i in [0, 1, 10, 100, 1000, 10_000, 100_000] {
            let alpha = alpha_for(i);
            assert!(alpha.is_finite());
            assert!(alpha > 0.0);
        }
    }
}
