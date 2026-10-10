//! psf_zero_core57 -- item 57b (candidate 2026-10-10.c30): the per-gate loops of `excitation_cost`,
//! `hybrid_cost` (its gate terms) and item 39's `_apply_ops`, for psf_compile.py. A module of its own, so that the
//! release's core (psf_zero_core) is not rebuilt or changed while the candidate is tested.
//!
//! Each function takes one byte buffer (the format is `statevec::parse`'s) and releases the GIL while it computes.

mod statevec;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use statevec::Kind;

fn bad(e: statevec::Error) -> PyErr {
    PyValueError::new_err(format!("psf_zero_core57: malformed buffer: {}", e))
}

/// The estimate the buffer asks for (kind 0: `excitation_cost`; kind 1: `hybrid_cost` without `readout_cost`).
#[pyfunction]
fn estimate57(py: Python<'_>, buf: &PyBytes) -> PyResult<f64> {
    let (kind, problem, _) = statevec::parse(buf.as_bytes()).map_err(bad)?;
    match kind {
        Kind::Excitation => Ok(py.allow_threads(|| statevec::excitation_cost(&problem))),
        Kind::HybridGates => Ok(py.allow_threads(|| statevec::hybrid_cost_gates(&problem))),
        Kind::ApplyOps => Err(PyValueError::new_err("psf_zero_core57: estimate57 got an apply_ops buffer")),
    }
}

/// `_apply_ops` (kind 2): the state after the ops, as interleaved little-endian f64 bytes.
#[pyfunction]
fn apply_ops57<'py>(py: Python<'py>, buf: &PyBytes) -> PyResult<&'py PyBytes> {
    let (kind, problem, state) = statevec::parse(buf.as_bytes()).map_err(bad)?;
    let mut state = match (kind, state) {
        (Kind::ApplyOps, Some(s)) => s,
        _ => return Err(PyValueError::new_err("psf_zero_core57: apply_ops57 needs an apply_ops buffer")),
    };
    let out = py.allow_threads(|| {
        statevec::apply_ops(&mut state, problem.k, &problem.ops);
        statevec::state_bytes(&state)
    });
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn psf_zero_core57(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate57, m)?)?;
    m.add_function(wrap_pyfunction!(apply_ops57, m)?)?;
    m.add("CORE57_VERSION", "2026-10-10.c30")?;
    m.add("BUFFER_VERSION", statevec::VERSION)?;
    Ok(())
}
