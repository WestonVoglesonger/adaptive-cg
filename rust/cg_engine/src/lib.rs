use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

/// Boltzmann constant in kJ/(mol*K)
const KB: f64 = 0.008314462618;

// ---------------------------------------------------------------------------
// Force computation
// ---------------------------------------------------------------------------

/// Compute harmonic bond forces and energy.
/// U = k * (r - r0)^2
/// Returns (forces [n,3], energy)
#[pyfunction]
fn compute_bond_forces<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    bond_i: PyReadonlyArray1<'py, i64>,
    bond_j: PyReadonlyArray1<'py, i64>,
    bond_r0: PyReadonlyArray1<'py, f64>,
    bond_k: PyReadonlyArray1<'py, f64>,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let pos = positions.as_array();
    let bi = bond_i.as_array();
    let bj = bond_j.as_array();
    let r0 = bond_r0.as_array();
    let k = bond_k.as_array();
    let n = pos.shape()[0];
    let n_bonds = bi.len();

    let mut forces = Array2::<f64>::zeros((n, 3));
    let mut energy = 0.0;

    for b in 0..n_bonds {
        let i = bi[b] as usize;
        let j = bj[b] as usize;

        let dx = pos[[j, 0]] - pos[[i, 0]];
        let dy = pos[[j, 1]] - pos[[i, 1]];
        let dz = pos[[j, 2]] - pos[[i, 2]];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        if r < 1e-12 {
            continue;
        }

        let inv_r = 1.0 / r;
        let dr = r - r0[b];
        let f_mag = -2.0 * k[b] * dr;

        let fx = f_mag * dx * inv_r;
        let fy = f_mag * dy * inv_r;
        let fz = f_mag * dz * inv_r;

        forces[[i, 0]] -= fx;
        forces[[i, 1]] -= fy;
        forces[[i, 2]] -= fz;
        forces[[j, 0]] += fx;
        forces[[j, 1]] += fy;
        forces[[j, 2]] += fz;

        energy += k[b] * dr * dr;
    }

    (forces.into_pyarray(py), energy)
}

/// Compute harmonic angle forces and energy.
/// For angle i-j-k (j is central): U = k * (theta - theta0)^2
#[pyfunction]
fn compute_angle_forces<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    angle_i: PyReadonlyArray1<'py, i64>,
    angle_j: PyReadonlyArray1<'py, i64>,
    angle_k: PyReadonlyArray1<'py, i64>,
    angle_theta0: PyReadonlyArray1<'py, f64>,
    angle_k_param: PyReadonlyArray1<'py, f64>,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let pos = positions.as_array();
    let ai = angle_i.as_array();
    let aj = angle_j.as_array();
    let ak = angle_k.as_array();
    let theta0 = angle_theta0.as_array();
    let k_param = angle_k_param.as_array();
    let n = pos.shape()[0];
    let n_angles = ai.len();

    let mut forces = Array2::<f64>::zeros((n, 3));
    let mut energy = 0.0;

    for a in 0..n_angles {
        let i = ai[a] as usize;
        let j = aj[a] as usize;
        let k = ak[a] as usize;

        // Vectors from central bead j
        let v1x = pos[[i, 0]] - pos[[j, 0]];
        let v1y = pos[[i, 1]] - pos[[j, 1]];
        let v1z = pos[[i, 2]] - pos[[j, 2]];
        let v2x = pos[[k, 0]] - pos[[j, 0]];
        let v2y = pos[[k, 1]] - pos[[j, 1]];
        let v2z = pos[[k, 2]] - pos[[j, 2]];

        let r1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
        let r2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();

        if r1 < 1e-12 || r2 < 1e-12 {
            continue;
        }

        let dot = v1x * v2x + v1y * v2y + v1z * v2z;
        let cos_theta = (dot / (r1 * r2)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let sin_theta = theta.sin();

        if sin_theta.abs() < 1e-12 {
            continue;
        }

        let dtheta = theta - theta0[a];
        energy += k_param[a] * dtheta * dtheta;

        let coeff = -2.0 * k_param[a] * dtheta / sin_theta;

        // Forces on i
        let fix = coeff * (cos_theta * v1x / r1 - v2x / r2) / r1;
        let fiy = coeff * (cos_theta * v1y / r1 - v2y / r2) / r1;
        let fiz = coeff * (cos_theta * v1z / r1 - v2z / r2) / r1;

        // Forces on k
        let fkx = coeff * (cos_theta * v2x / r2 - v1x / r1) / r2;
        let fky = coeff * (cos_theta * v2y / r2 - v1y / r1) / r2;
        let fkz = coeff * (cos_theta * v2z / r2 - v1z / r1) / r2;

        forces[[i, 0]] += fix;
        forces[[i, 1]] += fiy;
        forces[[i, 2]] += fiz;
        forces[[k, 0]] += fkx;
        forces[[k, 1]] += fky;
        forces[[k, 2]] += fkz;
        forces[[j, 0]] -= fix + fkx;
        forces[[j, 1]] -= fiy + fky;
        forces[[j, 2]] -= fiz + fkz;
    }

    (forces.into_pyarray(py), energy)
}

/// Compute Lennard-Jones non-bonded forces and energy.
/// U = 4*eps * [(sig/r)^12 - (sig/r)^6]
/// Includes soft minimum distance (0.5*sigma) and cutoff.
#[pyfunction]
fn compute_nonbonded_forces<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    pair_i: PyReadonlyArray1<'py, i64>,
    pair_j: PyReadonlyArray1<'py, i64>,
    pair_sigma: PyReadonlyArray1<'py, f64>,
    pair_epsilon: PyReadonlyArray1<'py, f64>,
    cutoff: f64,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let pos = positions.as_array();
    let pi = pair_i.as_array();
    let pj = pair_j.as_array();
    let sigma = pair_sigma.as_array();
    let epsilon = pair_epsilon.as_array();
    let n = pos.shape()[0];
    let n_pairs = pi.len();

    let mut forces = Array2::<f64>::zeros((n, 3));
    let mut energy = 0.0;

    for p in 0..n_pairs {
        let i = pi[p] as usize;
        let j = pj[p] as usize;

        let dx = pos[[j, 0]] - pos[[i, 0]];
        let dy = pos[[j, 1]] - pos[[i, 1]];
        let dz = pos[[j, 2]] - pos[[i, 2]];
        let mut r = (dx * dx + dy * dy + dz * dz).sqrt();

        if r < 1e-12 || r > cutoff {
            continue;
        }

        // Soft minimum distance
        let r_min = 0.5 * sigma[p];
        if r < r_min {
            r = r_min;
        }

        let inv_r = 1.0 / r;
        let sig_r = sigma[p] * inv_r;
        let sig_r6 = sig_r * sig_r * sig_r * sig_r * sig_r * sig_r;
        let sig_r12 = sig_r6 * sig_r6;

        energy += 4.0 * epsilon[p] * (sig_r12 - sig_r6);

        let f_mag = 24.0 * epsilon[p] * inv_r * (2.0 * sig_r12 - sig_r6);

        // Use original displacement direction (not clamped r)
        let r_orig = (dx * dx + dy * dy + dz * dz).sqrt();
        let inv_r_orig = if r_orig > 1e-12 { 1.0 / r_orig } else { 0.0 };

        let fx = f_mag * dx * inv_r_orig;
        let fy = f_mag * dy * inv_r_orig;
        let fz = f_mag * dz * inv_r_orig;

        forces[[i, 0]] -= fx;
        forces[[i, 1]] -= fy;
        forces[[i, 2]] -= fz;
        forces[[j, 0]] += fx;
        forces[[j, 1]] += fy;
        forces[[j, 2]] += fz;
    }

    (forces.into_pyarray(py), energy)
}

// ---------------------------------------------------------------------------
// Dihedral force computation (internal helper)
// ---------------------------------------------------------------------------

/// Compute periodic dihedral forces: U = k * (1 + cos(n*phi - phi0))
/// Adds forces to the provided array and returns energy contribution.
fn compute_dihedrals_inner(
    pos: &numpy::ndarray::ArrayView2<f64>,
    forces: &mut Array2<f64>,
    di: &numpy::ndarray::ArrayView1<i64>,
    dj: &numpy::ndarray::ArrayView1<i64>,
    dk: &numpy::ndarray::ArrayView1<i64>,
    dl: &numpy::ndarray::ArrayView1<i64>,
    dphi0: &numpy::ndarray::ArrayView1<f64>,
    dk_param: &numpy::ndarray::ArrayView1<f64>,
    dn: &numpy::ndarray::ArrayView1<i64>,
) -> f64 {
    let mut energy = 0.0;
    let n_dihedrals = di.len();

    for d in 0..n_dihedrals {
        let i = di[d] as usize;
        let j = dj[d] as usize;
        let k = dk[d] as usize;
        let l = dl[d] as usize;

        // Vectors along the chain
        let b1 = [pos[[j,0]]-pos[[i,0]], pos[[j,1]]-pos[[i,1]], pos[[j,2]]-pos[[i,2]]];
        let b2 = [pos[[k,0]]-pos[[j,0]], pos[[k,1]]-pos[[j,1]], pos[[k,2]]-pos[[j,2]]];
        let b3 = [pos[[l,0]]-pos[[k,0]], pos[[l,1]]-pos[[k,1]], pos[[l,2]]-pos[[k,2]]];

        // Normal vectors to planes
        let n1 = cross(&b1, &b2);
        let n2 = cross(&b2, &b3);
        let n1_len = norm(&n1);
        let n2_len = norm(&n2);
        if n1_len < 1e-12 || n2_len < 1e-12 { continue; }

        let n1_inv = 1.0 / n1_len;
        let n2_inv = 1.0 / n2_len;
        let n1u = [n1[0]*n1_inv, n1[1]*n1_inv, n1[2]*n1_inv];
        let n2u = [n2[0]*n2_inv, n2[1]*n2_inv, n2[2]*n2_inv];

        let cos_phi = dot(&n1u, &n2u).clamp(-1.0, 1.0);
        let sign = dot(&n1u, &b3).signum();
        let phi = sign * cos_phi.acos();

        let mult = dn[d] as f64;
        let k_val = dk_param[d];
        let phi0 = dphi0[d];

        energy += k_val * (1.0 + (mult * phi - phi0).cos());

        // dU/dphi = k * n * sin(n*phi - phi0)
        let du_dphi = k_val * mult * (mult * phi - phi0).sin();

        // Project torque onto atoms using standard dihedral force projection
        let b2_len = norm(&b2);
        if b2_len < 1e-12 { continue; }
        let b2_inv = 1.0 / b2_len;

        // Forces on i and l (terminal atoms)
        let fi_scale = -du_dphi / (n1_len * n1_len) * b2_len;
        let fl_scale = du_dphi / (n2_len * n2_len) * b2_len;

        for c in 0..3 {
            forces[[i, c]] += fi_scale * n1[c];
            forces[[l, c]] += fl_scale * n2[c];
        }

        // Forces on j and k (central atoms) — distribute to maintain zero net force
        let dot_b1_b2 = dot(&b1, &b2) * b2_inv * b2_inv;
        let dot_b3_b2 = dot(&b3, &b2) * b2_inv * b2_inv;

        for c in 0..3 {
            let fj = -fi_scale * n1[c] * (1.0 - dot_b1_b2) + fl_scale * n2[c] * dot_b3_b2;
            let fk = fi_scale * n1[c] * dot_b1_b2 - fl_scale * n2[c] * (1.0 - dot_b3_b2);
            forces[[j, c]] += fj;
            forces[[k, c]] += fk;
        }
    }
    energy
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

#[inline]
fn norm(a: &[f64; 3]) -> f64 {
    (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt()
}

// ---------------------------------------------------------------------------
// Combined force computation
// ---------------------------------------------------------------------------

/// Compute all forces (bond + angle + dihedral + nonbonded) in one call.
/// Returns (total_forces [n,3], bond_e, angle_e, dihedral_e, nb_e, potential_e)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_all_forces<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    // Bonds
    bond_i: PyReadonlyArray1<'py, i64>,
    bond_j: PyReadonlyArray1<'py, i64>,
    bond_r0: PyReadonlyArray1<'py, f64>,
    bond_k: PyReadonlyArray1<'py, f64>,
    // Angles
    angle_i: PyReadonlyArray1<'py, i64>,
    angle_j: PyReadonlyArray1<'py, i64>,
    angle_k: PyReadonlyArray1<'py, i64>,
    angle_theta0: PyReadonlyArray1<'py, f64>,
    angle_k_param: PyReadonlyArray1<'py, f64>,
    // Dihedrals
    dih_i: PyReadonlyArray1<'py, i64>,
    dih_j: PyReadonlyArray1<'py, i64>,
    dih_k: PyReadonlyArray1<'py, i64>,
    dih_l: PyReadonlyArray1<'py, i64>,
    dih_phi0: PyReadonlyArray1<'py, f64>,
    dih_k_param: PyReadonlyArray1<'py, f64>,
    dih_n: PyReadonlyArray1<'py, i64>,
    // Non-bonded
    pair_i: PyReadonlyArray1<'py, i64>,
    pair_j: PyReadonlyArray1<'py, i64>,
    pair_sigma: PyReadonlyArray1<'py, f64>,
    pair_epsilon: PyReadonlyArray1<'py, f64>,
    cutoff: f64,
) -> (Bound<'py, PyArray2<f64>>, f64, f64, f64, f64, f64) {
    let pos = positions.as_array();
    let n = pos.shape()[0];

    let mut forces = Array2::<f64>::zeros((n, 3));
    let mut e_bond = 0.0;
    let mut e_angle = 0.0;
    let mut e_nb = 0.0;

    // --- Bonds ---
    {
        let bi = bond_i.as_array();
        let bj = bond_j.as_array();
        let r0 = bond_r0.as_array();
        let k = bond_k.as_array();
        for b in 0..bi.len() {
            let i = bi[b] as usize;
            let j = bj[b] as usize;
            let dx = pos[[j, 0]] - pos[[i, 0]];
            let dy = pos[[j, 1]] - pos[[i, 1]];
            let dz = pos[[j, 2]] - pos[[i, 2]];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-12 { continue; }
            let inv_r = 1.0 / r;
            let dr = r - r0[b];
            let f_mag = -2.0 * k[b] * dr;
            let fx = f_mag * dx * inv_r;
            let fy = f_mag * dy * inv_r;
            let fz = f_mag * dz * inv_r;
            forces[[i, 0]] -= fx; forces[[i, 1]] -= fy; forces[[i, 2]] -= fz;
            forces[[j, 0]] += fx; forces[[j, 1]] += fy; forces[[j, 2]] += fz;
            e_bond += k[b] * dr * dr;
        }
    }

    // --- Angles ---
    {
        let ai = angle_i.as_array();
        let aj = angle_j.as_array();
        let ak = angle_k.as_array();
        let t0 = angle_theta0.as_array();
        let kp = angle_k_param.as_array();
        for a in 0..ai.len() {
            let i = ai[a] as usize;
            let j = aj[a] as usize;
            let k = ak[a] as usize;
            let v1x = pos[[i, 0]] - pos[[j, 0]];
            let v1y = pos[[i, 1]] - pos[[j, 1]];
            let v1z = pos[[i, 2]] - pos[[j, 2]];
            let v2x = pos[[k, 0]] - pos[[j, 0]];
            let v2y = pos[[k, 1]] - pos[[j, 1]];
            let v2z = pos[[k, 2]] - pos[[j, 2]];
            let r1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
            let r2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();
            if r1 < 1e-12 || r2 < 1e-12 { continue; }
            let cos_theta = ((v1x * v2x + v1y * v2y + v1z * v2z) / (r1 * r2)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();
            let sin_theta = theta.sin();
            if sin_theta.abs() < 1e-12 { continue; }
            let dtheta = theta - t0[a];
            e_angle += kp[a] * dtheta * dtheta;
            let coeff = -2.0 * kp[a] * dtheta / sin_theta;
            let fix = coeff * (cos_theta * v1x / r1 - v2x / r2) / r1;
            let fiy = coeff * (cos_theta * v1y / r1 - v2y / r2) / r1;
            let fiz = coeff * (cos_theta * v1z / r1 - v2z / r2) / r1;
            let fkx = coeff * (cos_theta * v2x / r2 - v1x / r1) / r2;
            let fky = coeff * (cos_theta * v2y / r2 - v1y / r1) / r2;
            let fkz = coeff * (cos_theta * v2z / r2 - v1z / r1) / r2;
            forces[[i, 0]] += fix; forces[[i, 1]] += fiy; forces[[i, 2]] += fiz;
            forces[[k, 0]] += fkx; forces[[k, 1]] += fky; forces[[k, 2]] += fkz;
            forces[[j, 0]] -= fix + fkx; forces[[j, 1]] -= fiy + fky; forces[[j, 2]] -= fiz + fkz;
        }
    }

    // --- Non-bonded ---
    {
        let pi_arr = pair_i.as_array();
        let pj_arr = pair_j.as_array();
        let sig = pair_sigma.as_array();
        let eps = pair_epsilon.as_array();
        for p in 0..pi_arr.len() {
            let i = pi_arr[p] as usize;
            let j = pj_arr[p] as usize;
            let dx = pos[[j, 0]] - pos[[i, 0]];
            let dy = pos[[j, 1]] - pos[[i, 1]];
            let dz = pos[[j, 2]] - pos[[i, 2]];
            let r_orig = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_orig < 1e-12 || r_orig > cutoff { continue; }
            let mut r = r_orig;
            let r_min = 0.5 * sig[p];
            if r < r_min { r = r_min; }
            let inv_r = 1.0 / r;
            let sig_r = sig[p] * inv_r;
            let sig_r3 = sig_r * sig_r * sig_r;
            let sig_r6 = sig_r3 * sig_r3;
            let sig_r12 = sig_r6 * sig_r6;
            e_nb += 4.0 * eps[p] * (sig_r12 - sig_r6);
            let f_mag = 24.0 * eps[p] * inv_r * (2.0 * sig_r12 - sig_r6);
            let inv_r_orig = 1.0 / r_orig;
            let fx = f_mag * dx * inv_r_orig;
            let fy = f_mag * dy * inv_r_orig;
            let fz = f_mag * dz * inv_r_orig;
            forces[[i, 0]] -= fx; forces[[i, 1]] -= fy; forces[[i, 2]] -= fz;
            forces[[j, 0]] += fx; forces[[j, 1]] += fy; forces[[j, 2]] += fz;
        }
    }

    // --- Dihedrals ---
    let e_dih = compute_dihedrals_inner(
        &pos, &mut forces,
        &dih_i.as_array(), &dih_j.as_array(),
        &dih_k.as_array(), &dih_l.as_array(),
        &dih_phi0.as_array(), &dih_k_param.as_array(),
        &dih_n.as_array(),
    );

    let e_total = e_bond + e_angle + e_dih + e_nb;
    (forces.into_pyarray(py), e_bond, e_angle, e_dih, e_nb, e_total)
}

// ---------------------------------------------------------------------------
// Integrator: BAOAB Langevin step
// ---------------------------------------------------------------------------

/// One BAOAB Langevin dynamics step. Modifies positions and velocities in-place.
/// Returns new forces after the step.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn langevin_step_rs<'py>(
    py: Python<'py>,
    // Mutable state (modified in-place via numpy)
    positions: &Bound<'py, PyArray2<f64>>,
    velocities: &Bound<'py, PyArray2<f64>>,
    masses: PyReadonlyArray1<'py, f64>,
    forces: PyReadonlyArray2<'py, f64>,
    dt: f64,
    temperature: f64,
    friction: f64,
    // Topology for force recomputation
    bond_i: PyReadonlyArray1<'py, i64>,
    bond_j: PyReadonlyArray1<'py, i64>,
    bond_r0: PyReadonlyArray1<'py, f64>,
    bond_k: PyReadonlyArray1<'py, f64>,
    angle_i: PyReadonlyArray1<'py, i64>,
    angle_j: PyReadonlyArray1<'py, i64>,
    angle_k: PyReadonlyArray1<'py, i64>,
    angle_theta0: PyReadonlyArray1<'py, f64>,
    angle_k_param: PyReadonlyArray1<'py, f64>,
    dih_i: PyReadonlyArray1<'py, i64>,
    dih_j: PyReadonlyArray1<'py, i64>,
    dih_k: PyReadonlyArray1<'py, i64>,
    dih_l: PyReadonlyArray1<'py, i64>,
    dih_phi0: PyReadonlyArray1<'py, f64>,
    dih_k_param: PyReadonlyArray1<'py, f64>,
    dih_n: PyReadonlyArray1<'py, i64>,
    pair_i: PyReadonlyArray1<'py, i64>,
    pair_j: PyReadonlyArray1<'py, i64>,
    pair_sigma: PyReadonlyArray1<'py, f64>,
    pair_epsilon: PyReadonlyArray1<'py, f64>,
    cutoff: f64,
) -> (Bound<'py, PyArray2<f64>>, f64, f64, f64, f64, f64) {
    let m = masses.as_array();
    let f = forces.as_array();
    let half_dt = 0.5 * dt;

    // SAFETY: we need mutable access to positions and velocities
    let mut pos = unsafe { positions.as_array_mut() };
    let mut vel = unsafe { velocities.as_array_mut() };
    let n = pos.shape()[0];

    // B: half-step velocity from forces
    for i in 0..n {
        let inv_m = 1.0 / m[i];
        vel[[i, 0]] += half_dt * f[[i, 0]] * inv_m;
        vel[[i, 1]] += half_dt * f[[i, 1]] * inv_m;
        vel[[i, 2]] += half_dt * f[[i, 2]] * inv_m;
    }

    // A: half-step position
    for i in 0..n {
        pos[[i, 0]] += half_dt * vel[[i, 0]];
        pos[[i, 1]] += half_dt * vel[[i, 1]];
        pos[[i, 2]] += half_dt * vel[[i, 2]];
    }

    // O: Ornstein-Uhlenbeck thermostat
    let c1 = (-friction * dt).exp();
    let c2 = ((1.0 - c1 * c1) * KB * temperature).sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    for i in 0..n {
        let sigma = c2 / m[i].sqrt();
        vel[[i, 0]] = c1 * vel[[i, 0]] + sigma * normal.sample(&mut rng);
        vel[[i, 1]] = c1 * vel[[i, 1]] + sigma * normal.sample(&mut rng);
        vel[[i, 2]] = c1 * vel[[i, 2]] + sigma * normal.sample(&mut rng);
    }

    // A: half-step position
    for i in 0..n {
        pos[[i, 0]] += half_dt * vel[[i, 0]];
        pos[[i, 1]] += half_dt * vel[[i, 1]];
        pos[[i, 2]] += half_dt * vel[[i, 2]];
    }

    // Compute new forces at updated positions
    // We need to read pos as immutable now — copy it
    let pos_copy = pos.to_owned();

    let mut new_forces = Array2::<f64>::zeros((n, 3));
    let mut e_bond = 0.0;
    let mut e_angle = 0.0;
    let mut e_nb = 0.0;

    // Bonds
    {
        let bi = bond_i.as_array();
        let bj = bond_j.as_array();
        let r0 = bond_r0.as_array();
        let kk = bond_k.as_array();
        for b in 0..bi.len() {
            let ii = bi[b] as usize;
            let jj = bj[b] as usize;
            let dx = pos_copy[[jj, 0]] - pos_copy[[ii, 0]];
            let dy = pos_copy[[jj, 1]] - pos_copy[[ii, 1]];
            let dz = pos_copy[[jj, 2]] - pos_copy[[ii, 2]];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-12 { continue; }
            let dr = r - r0[b];
            let f_mag = -2.0 * kk[b] * dr / r;
            new_forces[[ii, 0]] -= f_mag * dx; new_forces[[ii, 1]] -= f_mag * dy; new_forces[[ii, 2]] -= f_mag * dz;
            new_forces[[jj, 0]] += f_mag * dx; new_forces[[jj, 1]] += f_mag * dy; new_forces[[jj, 2]] += f_mag * dz;
            e_bond += kk[b] * dr * dr;
        }
    }

    // Angles
    {
        let ai = angle_i.as_array();
        let aj = angle_j.as_array();
        let ak = angle_k.as_array();
        let t0 = angle_theta0.as_array();
        let kp = angle_k_param.as_array();
        for a in 0..ai.len() {
            let ii = ai[a] as usize;
            let jj = aj[a] as usize;
            let kk = ak[a] as usize;
            let v1x = pos_copy[[ii, 0]] - pos_copy[[jj, 0]];
            let v1y = pos_copy[[ii, 1]] - pos_copy[[jj, 1]];
            let v1z = pos_copy[[ii, 2]] - pos_copy[[jj, 2]];
            let v2x = pos_copy[[kk, 0]] - pos_copy[[jj, 0]];
            let v2y = pos_copy[[kk, 1]] - pos_copy[[jj, 1]];
            let v2z = pos_copy[[kk, 2]] - pos_copy[[jj, 2]];
            let r1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
            let r2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();
            if r1 < 1e-12 || r2 < 1e-12 { continue; }
            let cos_theta = ((v1x * v2x + v1y * v2y + v1z * v2z) / (r1 * r2)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();
            let sin_theta = theta.sin();
            if sin_theta.abs() < 1e-12 { continue; }
            let dtheta = theta - t0[a];
            e_angle += kp[a] * dtheta * dtheta;
            let coeff = -2.0 * kp[a] * dtheta / sin_theta;
            let fix = coeff * (cos_theta * v1x / r1 - v2x / r2) / r1;
            let fiy = coeff * (cos_theta * v1y / r1 - v2y / r2) / r1;
            let fiz = coeff * (cos_theta * v1z / r1 - v2z / r2) / r1;
            let fkx = coeff * (cos_theta * v2x / r2 - v1x / r1) / r2;
            let fky = coeff * (cos_theta * v2y / r2 - v1y / r1) / r2;
            let fkz = coeff * (cos_theta * v2z / r2 - v1z / r1) / r2;
            new_forces[[ii, 0]] += fix; new_forces[[ii, 1]] += fiy; new_forces[[ii, 2]] += fiz;
            new_forces[[kk, 0]] += fkx; new_forces[[kk, 1]] += fky; new_forces[[kk, 2]] += fkz;
            new_forces[[jj, 0]] -= fix + fkx; new_forces[[jj, 1]] -= fiy + fky; new_forces[[jj, 2]] -= fiz + fkz;
        }
    }

    // Non-bonded
    {
        let pi_arr = pair_i.as_array();
        let pj_arr = pair_j.as_array();
        let sig = pair_sigma.as_array();
        let eps = pair_epsilon.as_array();
        for p in 0..pi_arr.len() {
            let ii = pi_arr[p] as usize;
            let jj = pj_arr[p] as usize;
            let dx = pos_copy[[jj, 0]] - pos_copy[[ii, 0]];
            let dy = pos_copy[[jj, 1]] - pos_copy[[ii, 1]];
            let dz = pos_copy[[jj, 2]] - pos_copy[[ii, 2]];
            let r_orig = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_orig < 1e-12 || r_orig > cutoff { continue; }
            let mut r = r_orig;
            let r_min = 0.5 * sig[p];
            if r < r_min { r = r_min; }
            let inv_r = 1.0 / r;
            let sig_r = sig[p] * inv_r;
            let sig_r3 = sig_r * sig_r * sig_r;
            let sig_r6 = sig_r3 * sig_r3;
            let sig_r12 = sig_r6 * sig_r6;
            e_nb += 4.0 * eps[p] * (sig_r12 - sig_r6);
            let f_mag = 24.0 * eps[p] * inv_r * (2.0 * sig_r12 - sig_r6);
            let inv_r_orig = 1.0 / r_orig;
            let fx = f_mag * dx * inv_r_orig;
            let fy = f_mag * dy * inv_r_orig;
            let fz = f_mag * dz * inv_r_orig;
            new_forces[[ii, 0]] -= fx; new_forces[[ii, 1]] -= fy; new_forces[[ii, 2]] -= fz;
            new_forces[[jj, 0]] += fx; new_forces[[jj, 1]] += fy; new_forces[[jj, 2]] += fz;
        }
    }

    // Dihedrals
    let e_dih = compute_dihedrals_inner(
        &pos_copy.view(), &mut new_forces,
        &dih_i.as_array(), &dih_j.as_array(),
        &dih_k.as_array(), &dih_l.as_array(),
        &dih_phi0.as_array(), &dih_k_param.as_array(),
        &dih_n.as_array(),
    );

    // B: half-step velocity from new forces
    for i in 0..n {
        let inv_m = 1.0 / m[i];
        vel[[i, 0]] += half_dt * new_forces[[i, 0]] * inv_m;
        vel[[i, 1]] += half_dt * new_forces[[i, 1]] * inv_m;
        vel[[i, 2]] += half_dt * new_forces[[i, 2]] * inv_m;
    }

    let e_total = e_bond + e_angle + e_dih + e_nb;
    (new_forces.into_pyarray(py), e_bond, e_angle, e_dih, e_nb, e_total)
}

// ---------------------------------------------------------------------------
// Energy minimization
// ---------------------------------------------------------------------------

/// Steepest descent energy minimization with force capping.
/// Returns final potential energy.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn minimize_energy_rs<'py>(
    _py: Python<'py>,
    positions: &Bound<'py, PyArray2<f64>>,
    // Topology
    bond_i: PyReadonlyArray1<'py, i64>,
    bond_j: PyReadonlyArray1<'py, i64>,
    bond_r0: PyReadonlyArray1<'py, f64>,
    bond_k: PyReadonlyArray1<'py, f64>,
    angle_i: PyReadonlyArray1<'py, i64>,
    angle_j: PyReadonlyArray1<'py, i64>,
    angle_k: PyReadonlyArray1<'py, i64>,
    angle_theta0: PyReadonlyArray1<'py, f64>,
    angle_k_param: PyReadonlyArray1<'py, f64>,
    dih_i: PyReadonlyArray1<'py, i64>,
    dih_j: PyReadonlyArray1<'py, i64>,
    dih_k: PyReadonlyArray1<'py, i64>,
    dih_l: PyReadonlyArray1<'py, i64>,
    dih_phi0: PyReadonlyArray1<'py, f64>,
    dih_k_param: PyReadonlyArray1<'py, f64>,
    dih_n: PyReadonlyArray1<'py, i64>,
    pair_i: PyReadonlyArray1<'py, i64>,
    pair_j: PyReadonlyArray1<'py, i64>,
    pair_sigma: PyReadonlyArray1<'py, f64>,
    pair_epsilon: PyReadonlyArray1<'py, f64>,
    cutoff: f64,
    max_steps: usize,
    step_size: f64,
    force_cap: f64,
    tolerance: f64,
) -> f64 {
    let mut pos = unsafe { positions.as_array_mut() };
    let n = pos.shape()[0];

    let bi = bond_i.as_array();
    let bj = bond_j.as_array();
    let br0 = bond_r0.as_array();
    let bk = bond_k.as_array();
    let ai = angle_i.as_array();
    let aj = angle_j.as_array();
    let ak = angle_k.as_array();
    let at0 = angle_theta0.as_array();
    let akp = angle_k_param.as_array();
    let pi_arr = pair_i.as_array();
    let pj_arr = pair_j.as_array();
    let sig = pair_sigma.as_array();
    let eps = pair_epsilon.as_array();

    let mut final_pe = 0.0;

    for _step in 0..max_steps {
        let mut forces = Array2::<f64>::zeros((n, 3));
        let mut pe = 0.0;

        // Bonds
        for b in 0..bi.len() {
            let i = bi[b] as usize;
            let j = bj[b] as usize;
            let dx = pos[[j, 0]] - pos[[i, 0]];
            let dy = pos[[j, 1]] - pos[[i, 1]];
            let dz = pos[[j, 2]] - pos[[i, 2]];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-12 { continue; }
            let dr = r - br0[b];
            let f_mag = -2.0 * bk[b] * dr / r;
            forces[[i, 0]] -= f_mag * dx; forces[[i, 1]] -= f_mag * dy; forces[[i, 2]] -= f_mag * dz;
            forces[[j, 0]] += f_mag * dx; forces[[j, 1]] += f_mag * dy; forces[[j, 2]] += f_mag * dz;
            pe += bk[b] * dr * dr;
        }

        // Angles
        for a in 0..ai.len() {
            let i = ai[a] as usize;
            let j = aj[a] as usize;
            let k = ak[a] as usize;
            let v1x = pos[[i, 0]] - pos[[j, 0]];
            let v1y = pos[[i, 1]] - pos[[j, 1]];
            let v1z = pos[[i, 2]] - pos[[j, 2]];
            let v2x = pos[[k, 0]] - pos[[j, 0]];
            let v2y = pos[[k, 1]] - pos[[j, 1]];
            let v2z = pos[[k, 2]] - pos[[j, 2]];
            let r1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
            let r2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();
            if r1 < 1e-12 || r2 < 1e-12 { continue; }
            let cos_theta = ((v1x * v2x + v1y * v2y + v1z * v2z) / (r1 * r2)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();
            let sin_theta = theta.sin();
            if sin_theta.abs() < 1e-12 { continue; }
            let dtheta = theta - at0[a];
            pe += akp[a] * dtheta * dtheta;
            let coeff = -2.0 * akp[a] * dtheta / sin_theta;
            let fix = coeff * (cos_theta * v1x / r1 - v2x / r2) / r1;
            let fiy = coeff * (cos_theta * v1y / r1 - v2y / r2) / r1;
            let fiz = coeff * (cos_theta * v1z / r1 - v2z / r2) / r1;
            let fkx = coeff * (cos_theta * v2x / r2 - v1x / r1) / r2;
            let fky = coeff * (cos_theta * v2y / r2 - v1y / r1) / r2;
            let fkz = coeff * (cos_theta * v2z / r2 - v1z / r1) / r2;
            forces[[i, 0]] += fix; forces[[i, 1]] += fiy; forces[[i, 2]] += fiz;
            forces[[k, 0]] += fkx; forces[[k, 1]] += fky; forces[[k, 2]] += fkz;
            forces[[j, 0]] -= fix + fkx; forces[[j, 1]] -= fiy + fky; forces[[j, 2]] -= fiz + fkz;
        }

        // Non-bonded
        for p in 0..pi_arr.len() {
            let i = pi_arr[p] as usize;
            let j = pj_arr[p] as usize;
            let dx = pos[[j, 0]] - pos[[i, 0]];
            let dy = pos[[j, 1]] - pos[[i, 1]];
            let dz = pos[[j, 2]] - pos[[i, 2]];
            let r_orig = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_orig < 1e-12 || r_orig > cutoff { continue; }
            let mut r = r_orig;
            let r_min = 0.5 * sig[p];
            if r < r_min { r = r_min; }
            let inv_r = 1.0 / r;
            let sig_r = sig[p] * inv_r;
            let sig_r3 = sig_r * sig_r * sig_r;
            let sig_r6 = sig_r3 * sig_r3;
            let sig_r12 = sig_r6 * sig_r6;
            pe += 4.0 * eps[p] * (sig_r12 - sig_r6);
            let f_mag = 24.0 * eps[p] * inv_r * (2.0 * sig_r12 - sig_r6);
            let inv_r_orig = 1.0 / r_orig;
            let fx = f_mag * dx * inv_r_orig;
            let fy = f_mag * dy * inv_r_orig;
            let fz = f_mag * dz * inv_r_orig;
            forces[[i, 0]] -= fx; forces[[i, 1]] -= fy; forces[[i, 2]] -= fz;
            forces[[j, 0]] += fx; forces[[j, 1]] += fy; forces[[j, 2]] += fz;
        }

        // Dihedrals
        let pos_view = pos.view();
        pe += compute_dihedrals_inner(
            &pos_view, &mut forces,
            &dih_i.as_array(), &dih_j.as_array(),
            &dih_k.as_array(), &dih_l.as_array(),
            &dih_phi0.as_array(), &dih_k_param.as_array(),
            &dih_n.as_array(),
        );

        // Check convergence and move
        let mut max_f = 0.0_f64;
        for i in 0..n {
            let fn2 = forces[[i, 0]] * forces[[i, 0]]
                + forces[[i, 1]] * forces[[i, 1]]
                + forces[[i, 2]] * forces[[i, 2]];
            let fnorm = fn2.sqrt();
            if fnorm > max_f {
                max_f = fnorm;
            }
        }

        if max_f < tolerance {
            return pe;
        }

        // Steepest descent: normalized step per bead
        for i in 0..n {
            let fn2 = forces[[i, 0]] * forces[[i, 0]]
                + forces[[i, 1]] * forces[[i, 1]]
                + forces[[i, 2]] * forces[[i, 2]];
            let fnorm = fn2.sqrt();
            if fnorm < 1e-12 { continue; }

            let scale = if fnorm > force_cap {
                step_size * force_cap / fnorm
            } else {
                step_size
            };

            let inv_fn = 1.0 / fnorm;
            pos[[i, 0]] += scale * forces[[i, 0]] * inv_fn;
            pos[[i, 1]] += scale * forces[[i, 1]] * inv_fn;
            pos[[i, 2]] += scale * forces[[i, 2]] * inv_fn;
        }

        final_pe = pe;
    }

    final_pe
}

// ---------------------------------------------------------------------------
// Collision detection
// ---------------------------------------------------------------------------

/// Detect dangerous close contacts between non-bonded bead pairs.
/// Returns list of (pair_index, bead_i, bead_j, distance, sigma, epsilon)
/// for all pairs where distance < danger_fraction * sigma.
#[pyfunction]
fn detect_collisions<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    pair_i: PyReadonlyArray1<'py, i64>,
    pair_j: PyReadonlyArray1<'py, i64>,
    pair_sigma: PyReadonlyArray1<'py, f64>,
    pair_epsilon: PyReadonlyArray1<'py, f64>,
    danger_fraction: f64,
) -> Vec<(usize, i64, i64, f64, f64, f64)> {
    let pos = positions.as_array();
    let pi = pair_i.as_array();
    let pj = pair_j.as_array();
    let sig = pair_sigma.as_array();
    let eps = pair_epsilon.as_array();

    let mut collisions = Vec::new();
    for p in 0..pi.len() {
        let i = pi[p] as usize;
        let j = pj[p] as usize;
        let dx = pos[[j, 0]] - pos[[i, 0]];
        let dy = pos[[j, 1]] - pos[[i, 1]];
        let dz = pos[[j, 2]] - pos[[i, 2]];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < danger_fraction * sig[p] && r > 1e-12 {
            collisions.push((p, pi[p], pj[p], r, sig[p], eps[p]));
        }
    }
    collisions
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn cg_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_bond_forces, m)?)?;
    m.add_function(wrap_pyfunction!(compute_angle_forces, m)?)?;
    m.add_function(wrap_pyfunction!(compute_nonbonded_forces, m)?)?;
    m.add_function(wrap_pyfunction!(compute_all_forces, m)?)?;
    m.add_function(wrap_pyfunction!(langevin_step_rs, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_energy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(detect_collisions, m)?)?;
    Ok(())
}
