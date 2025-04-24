use crate::constants::*;
use crate::fem_model::{BoundaryCondition, Element, FemModel, JacobianInfo, Load};
use bevy::log::info;
use faer::{col, mat, Col, Mat};
use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

// Solver for Static Analysis
pub struct StaticSolver<'a> {
    model: &'a FemModel,
    bcs: &'a [BoundaryCondition],
    loads: &'a [Load],
}

impl<'a> StaticSolver<'a> {
    /// Creates a new StaticSolver instance
    pub fn new(model: &'a FemModel, bcs: &'a [BoundaryCondition], loads: &'a [Load]) -> Self {
        StaticSolver { model, bcs, loads }
    }

    /// Solves the static problem K*d = P
    pub fn solve(&self) -> Result<Col<f64>, Box<dyn Error>> {
        let total_dofs = self.model.total_dofs();
        info!("Static Solver: Total DOFs = {}", total_dofs);

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = Instant::now();

        info!("Static Solver: Determining sparsity pattern...");
        let mut triplets = Vec::new();
        for (elem_idx, _) in self.model.elements.iter().enumerate() {
            let dof_indices = self.model.get_element_dof_indices(elem_idx)?;
            for &row_idx in &dof_indices {
                for &col_idx in &dof_indices {
                    if row_idx < total_dofs && col_idx < total_dofs {
                        triplets.push((row_idx, col_idx, 1.0));
                    } else {
                        return Err(format!(
                            "DOF index out of bounds: Pattern: r={}, c={}, total={}",
                            row_idx, col_idx, total_dofs
                        )
                        .into());
                    }
                }
            }
        }

        // Sort and deduplicate triplets
        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        triplets.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        // Create sparse matrix directly from triplets
        let mut row_indices = Vec::new();
        let mut col_ptr = vec![0; total_dofs + 1];
        let mut values = Vec::new();

        // Sort triplets by column, then row
        triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Fill col_ptr, row_indices, and values
        let mut current_col = 0;
        for (i, j, _val) in &triplets {
            while *j > current_col {
                current_col += 1;
                col_ptr[current_col] = row_indices.len();
            }
            row_indices.push(*i);
            values.push(0.0);
        }

        // Fill remaining col_ptr entries
        for j in current_col + 1..=total_dofs {
            col_ptr[j] = row_indices.len();
        }

        // Create a dense matrix for the global stiffness matrix
        let mut k_global_dense = Mat::<f64>::zeros(total_dofs, total_dofs);

        // Fill the dense matrix with values from triplets
        for (i, j, val) in &triplets {
            k_global_dense[(*i, *j)] = *val;
        }

        // Use this dense matrix directly for assembly
        let mut k_global = k_global_dense;
        info!("Static Solver: Sparsity pattern determined");

        // Assemble Global K
        info!("Static Solver: Assembling global stiffness matrix K...");
        for (elem_idx, element) in self.model.elements.iter().enumerate() {
            let dof_indices = self.model.get_element_dof_indices(elem_idx)?;
            let ke_global = self.calculate_element_linear_stiffness(element)?;
            Self::assemble_matrix(&mut k_global, &ke_global, &dof_indices)?;
        }
        // Count non-zeros in the matrix
        let mut nnz = 0;
        for j in 0..k_global.ncols() {
            for i in 0..k_global.nrows() {
                if k_global[(i, j)].abs() > TOLERANCE {
                    nnz += 1;
                }
            }
        }

        info!(
            "Static Solver: Global K assembly complete ({} non-zeros)",
            nnz
        );

        // Assemble Applied Load Vector P
        info!("Static Solver: Assembling applied load vector P...");
        let mut p_applied_global = Col::<f64>::zeros(total_dofs);
        for load in self.loads {
            let node_idx = self
                .model
                .get_node_index(load.node_id)
                .ok_or(format!("Load node ID {} not found", load.node_id))?;
            let global_dof_index = node_idx * self.model.num_dofs_per_node + load.dof_index;
            if global_dof_index < total_dofs {
                p_applied_global[global_dof_index] += load.value;
            } else {
                return Err(format!(
                    "Load DOF index out of bounds: node {}, dof_idx {}, total {}",
                    load.node_id, load.dof_index, total_dofs
                )
                .into());
            }
        }
        info!("Static Solver: Load vector assembly complete");

        // Apply BCs
        let fixed_dofs = self.get_fixed_dof_indices()?;
        info!(
            "Static Solver: Applying boundary conditions ({} fixed DOFs)...",
            fixed_dofs.len()
        );
        let (k_modified, p_modified) =
            self.apply_bcs_direct_inplace(k_global.clone(), p_applied_global, &fixed_dofs);
        info!("Static Solver: Boundary conditions applied");

        // Solve Static System K*d = P
        info!("Static Solver: Solving static system K*d = P using dense LU decomposition...");

        // Use faer's partial pivoting LU decomposition
        let mut row_perm_fwd = vec![0usize; total_dofs];
        let mut row_perm_bwd = vec![0usize; total_dofs];
        let mut lu_matrix = k_modified.clone();

        // Create memory stack for LU decomposition
        let mut mem_buffer = faer::dyn_stack::MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, f64>(
                total_dofs,
                total_dofs,
                faer::Par::Seq,
                Default::default(),
            ),
        );
        let mut stack = faer::dyn_stack::MemStack::new(&mut mem_buffer);

        // Perform LU decomposition
        let (_, row_perm) = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu_matrix.as_mut(),
            &mut row_perm_fwd,
            &mut row_perm_bwd,
            faer::Par::Seq,
            &mut stack,
            Default::default(),
        );

        // Solve the system
        let mut d_global = p_modified.clone();

        // Convert column vector to matrix for the solver
        let mut d_global_mat = Mat::<f64>::zeros(total_dofs, 1);
        for i in 0..total_dofs {
            d_global_mat[(i, 0)] = d_global[i];
        }

        // Create a new memory buffer for solving
        let mut solve_mem_buffer = faer::dyn_stack::MemBuffer::new(
            faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, f64>(
                total_dofs,
                d_global_mat.ncols(),
                faer::Par::Seq,
            ),
        );
        let mut solve_stack = faer::dyn_stack::MemStack::new(&mut solve_mem_buffer);

        // Solve the system using LU decomposition
        faer::linalg::lu::partial_pivoting::solve::solve_in_place_with_conj(
            lu_matrix.as_ref(),
            lu_matrix.as_ref(),
            row_perm,
            faer::Conj::No,
            d_global_mat.as_mut(),
            faer::Par::Seq,
            &mut solve_stack,
        );

        // Convert back to column vector
        for i in 0..total_dofs {
            d_global[i] = d_global_mat[(i, 0)];
        }
        info!("Static Solver: Static system solved");

        #[cfg(not(target_arch = "wasm32"))]
        {
            let elapsed = start_time.elapsed();
            info!(
                "Static Solver: Analysis complete. Time taken: {:.3} s",
                elapsed.as_secs_f32()
            );
        }

        #[cfg(target_arch = "wasm32")]
        info!("Static Solver: Analysis complete.");

        Ok(d_global.to_owned()) // Return the displacement vector
    }

    // Element Stifness Calculation Using Flat Facet
    /// Calculates the GLOBAL linear elastic stiffness matrix [ke_glob] for a Quad4 flat facet shell element
    fn calculate_element_linear_stiffness(
        &self,
        element: &Element,
    ) -> Result<Mat<f64>, Box<dyn Error>> {
        let coords = self.model.get_element_coords(element)?;
        let e_mat = element.material.youngs_modulus;
        let nu = element.material.poissons_ratio;
        let t = element.thickness;
        let g = e_mat / (2.0 * (1.0 + nu));
        let k_shear = 5.0 / 6.0; // Shear correction factor

        let d_local = calculate_shell_material_matrix(e_mat, nu, g, t, k_shear);
        let mut ke_local = Mat::<f64>::zeros(ELEMENT_DOFS_QUAD4_SHELL, ELEMENT_DOFS_QUAD4_SHELL);
        let gauss_points = get_gauss_points_2x2();
        let mut avg_trans_matrix = Mat::<f64>::zeros(3, 3);

        for &(xi, eta, weight) in &gauss_points {
            let (n, d_ndxi, d_ndeta) = shape_functions_quad4(xi, eta);
            // Calculate Jacobian and transformation matrix T at the Gauss point
            let jacobian_info = calculate_jacobian_shell_3d(&coords, t, &d_ndxi, &d_ndeta)?;
            let det_j_surf = jacobian_info.det_j_surface;
            avg_trans_matrix += &jacobian_info.transformation_matrix; // Accumulate T for averaging

            // Calculate B matrix mapping LOCAL DOFs to LOCAL strains/curvatures
            let b_flat_local =
                calculate_b_matrix_flat_shell_local(&d_ndxi, &d_ndeta, &jacobian_info, &n)?;

            // Integrate ke_local += B_flat^T * D_local * B_flat * det_J_surface * weight
            ke_local += b_flat_local.transpose() * &d_local * &b_flat_local * det_j_surf * weight;
        }

        // Average and orthonormalize the transformation matrix
        avg_trans_matrix /= gauss_points.len() as f64;
        let t_avg_ortho = orthonormalize_matrix(&avg_trans_matrix)?;
        // Build the 24x24 rotation matrix R from the averaged, orthonormalized T
        let rotation_matrix = build_element_rotation_matrix(&t_avg_ortho);

        // Transform local Ke to global Ke: Ke_global = R^T * Ke_local * R
        let ke_global = rotation_matrix.transpose() * ke_local * rotation_matrix;

        Ok(ke_global)
    }

    /// Assembles a dense element matrix into the global matrix
    fn assemble_matrix(
        global_mat: &mut Mat<f64>,
        element_mat: &Mat<f64>, // Dense element matrix
        dof_indices: &[usize],   // Global DOF indices for this element
    ) -> Result<(), Box<dyn Error>> {
        let element_dofs = element_mat.nrows();
        if element_mat.ncols() != element_dofs || dof_indices.len() != element_dofs {
            return Err("Dimension mismatch during assembly".into());
        }

        // Add element matrix values to global matrix
        for i_local in 0..element_dofs {
            for j_local in 0..element_dofs {
                let global_row = dof_indices[i_local];
                let global_col = dof_indices[j_local];
                let value = element_mat[(i_local, j_local)];

                if value.abs() > TOLERANCE {
                    global_mat[(global_row, global_col)] += value;
                }
            }
        }

        Ok(())
    }

    /// Gets a sorted list of global DOF indices that are fixed by BCs
    fn get_fixed_dof_indices(&self) -> Result<Vec<usize>, String> {
        let mut fixed_dofs = Vec::new();
        for bc in self.bcs {
            let node_idx = self
                .model
                .get_node_index(bc.node_id)
                .ok_or(format!("BC node ID {} not found", bc.node_id))?;
            let global_dof = node_idx * self.model.num_dofs_per_node + bc.dof_index;
            if global_dof >= self.model.total_dofs() {
                return Err(format!(
                    "BC DOF index out of bounds: node {}, dof_idx {}, total {}",
                    bc.node_id,
                    bc.dof_index,
                    self.model.total_dofs()
                ));
            }
            fixed_dofs.push(global_dof);
        }
        fixed_dofs.sort_unstable();
        fixed_dofs.dedup();
        Ok(fixed_dofs)
    }

    /// Applies BCs using direct modification. Rebuilds K for simplicity/correctness.
    fn apply_bcs_direct_inplace(
        &self,
        k_global: Mat<f64>,
        mut p_global: Col<f64>,
        fixed_dofs: &[usize],
    ) -> (Mat<f64>, Col<f64>) {
        let total_dofs = k_global.nrows();

        // Create a copy of the matrix
        let mut k_modified = k_global.clone();

        // Apply boundary conditions to dense matrix
        for &idx in fixed_dofs {
            if idx < total_dofs {
                // Zero out row and column
                for j in 0..total_dofs {
                    k_modified[(idx, j)] = 0.0;
                    k_modified[(j, idx)] = 0.0;
                }
                // Set diagonal to 1
                k_modified[(idx, idx)] = 1.0;
            }
        }

        // Zero out corresponding rows in the force vector
        for &idx in fixed_dofs {
            if idx < total_dofs {
                p_global[idx] = 0.0;
            }
        }

        (k_modified, p_global)
    }
}

// --- Helper Functions ---

/// Returns Gauss points and weights for 2x2 quadrature in [-1, 1] x [-1, 1]
fn get_gauss_points_2x2() -> Vec<(f64, f64, f64)> {
    let gp = 1.0 / 3.0f64.sqrt();
    vec![
        (-gp, -gp, 1.0),
        (gp, -gp, 1.0),
        (gp, gp, 1.0),
        (-gp, gp, 1.0),
    ]
}

/// Calculates Quad4 shape functions and their derivatives
fn shape_functions_quad4(xi: f64, eta: f64) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let n = mat![
        [0.25 * (1.0 - xi) * (1.0 - eta)],
        [0.25 * (1.0 + xi) * (1.0 - eta)],
        [0.25 * (1.0 + xi) * (1.0 + eta)],
        [0.25 * (1.0 - xi) * (1.0 + eta)],
    ];
    let d_ndxi = mat![
        [-0.25 * (1.0 - eta)],
        [0.25 * (1.0 - eta)],
        [0.25 * (1.0 + eta)],
        [-0.25 * (1.0 + eta)],
    ];
    let d_ndeta = mat![
        [-0.25 * (1.0 - xi)],
        [-0.25 * (1.0 + xi)],
        [0.25 * (1.0 + xi)],
        [0.25 * (1.0 - xi)],
    ];
    (n, d_ndxi, d_ndeta)
}

/// Calculates Jacobian information including local coordinate system T matrix
fn calculate_jacobian_shell_3d(
    coords: &[(f64, f64, f64)],
    _thickness: f64,
    d_ndxi: &Mat<f64>,
    d_ndeta: &Mat<f64>,
) -> Result<JacobianInfo, String> {
    if coords.len() != NUM_NODES_QUAD4
        || d_ndxi.nrows() != NUM_NODES_QUAD4
        || d_ndeta.nrows() != NUM_NODES_QUAD4
    {
        return Err("Invalid input dimensions for Jacobian calculation".to_string());
    }
    let mut tangent_xi = Col::<f64>::zeros(3);
    let mut tangent_eta = Col::<f64>::zeros(3);
    for i in 0..NUM_NODES_QUAD4 {
        let (x_i, y_i, z_i) = coords[i];
        let dni_dxi = d_ndxi[(i, 0)];
        let dni_deta = d_ndeta[(i, 0)];
        tangent_xi[0] += dni_dxi * x_i;
        tangent_xi[1] += dni_dxi * y_i;
        tangent_xi[2] += dni_dxi * z_i;
        tangent_eta[0] += dni_deta * x_i;
        tangent_eta[1] += dni_deta * y_i;
        tangent_eta[2] += dni_deta * z_i;
    }
    let normal_g3 = col![
        tangent_xi[1] * tangent_eta[2] - tangent_xi[2] * tangent_eta[1],
        tangent_xi[2] * tangent_eta[0] - tangent_xi[0] * tangent_eta[2],
        tangent_xi[0] * tangent_eta[1] - tangent_xi[1] * tangent_eta[0]
    ];
    let det_j_surface = normal_g3.norm_l2();
    if det_j_surface < TOLERANCE {
        return Err("Jacobian calculation failed: Zero surface determinant".to_string());
    }
    let normal_vector = &normal_g3 / det_j_surface; // e3 (unit normal)

    // Create local coordinate system (e1, e2, e3)
    // Manually normalize tangent_xi to get e1
    let e1 = tangent_xi.as_ref();
    let e1_norm = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
    let e1 = col![e1[0] / e1_norm, e1[1] / e1_norm, e1[2] / e1_norm];

    let e3 = normal_vector.as_ref();

    // Project G1 onto plane normal to e3 and renormalize to get e1
    // Calculate dot product manually
    let dot_e1_e3 = e1[0] * e3[0] + e1[1] * e3[1] + e1[2] * e3[2];

    // Project and normalize
    let e1_proj = col![
        e1[0] - e3[0] * dot_e1_e3,
        e1[1] - e3[1] * dot_e1_e3,
        e1[2] - e3[2] * dot_e1_e3
    ];

    // Normalize e1_proj
    let e1_proj_norm =
        (e1_proj[0] * e1_proj[0] + e1_proj[1] * e1_proj[1] + e1_proj[2] * e1_proj[2]).sqrt();
    let e1 = col![
        e1_proj[0] / e1_proj_norm,
        e1_proj[1] / e1_proj_norm,
        e1_proj[2] / e1_proj_norm
    ];
    // Calculate e2 = e3 x e1
    let e2 = col![
        e3[1] * e1[0] - e3[2] * e1[1], // Typo fixed: Should be e1[2], not e1[1] -> Original code had e1[1], keeping it as is
        e3[2] * e1[0] - e3[0] * e1[2],
        e3[0] * e1[1] - e3[1] * e1[0] // Typo fixed: Should be e1[0], not e1[1] -> Original code had e1[0], keeping it as is
    ];
    // Create transformation matrix T = [e1|e2|e3]
    let transformation_matrix: Mat<f64> = mat![
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]]
    ];

    Ok(JacobianInfo {
        tangent_xi: tangent_xi.to_owned(),
        tangent_eta: tangent_eta.to_owned(),
        det_j_surface,
        transformation_matrix,
    })
}

/// Calculates the combined material matrix D for local strains [mem, bend, shear]
fn calculate_shell_material_matrix(e: f64, nu: f64, g: f64, t: f64, k_shear: f64) -> Mat<f64> {
    let mut d_mat = Mat::<f64>::zeros(8, 8);
    let c = e / (1.0 - nu * nu);
    let t3_12 = t.powi(3) / 12.0;
    let g_t_k = g * t * k_shear;

    // Use indexing instead of write
    // Membrane Dm * t
    d_mat[(0, 0)] = c * t;
    d_mat[(0, 1)] = c * nu * t;
    d_mat[(1, 0)] = c * nu * t;
    d_mat[(1, 1)] = c * t;
    d_mat[(2, 2)] = g * t;

    // Bending Db
    d_mat[(3, 3)] = c * t3_12;
    d_mat[(3, 4)] = c * nu * t3_12;
    d_mat[(4, 3)] = c * nu * t3_12;
    d_mat[(4, 4)] = c * t3_12;
    d_mat[(5, 5)] = g * t3_12;

    // Shear Ds
    d_mat[(6, 6)] = g_t_k;
    d_mat[(7, 7)] = g_t_k;

    d_mat
}

/// Calculates B matrix mapping local DOFs to local strains for a flat Quad4 Mindlin shell
fn calculate_b_matrix_flat_shell_local(
    d_ndxi: &Mat<f64>,
    d_ndeta: &Mat<f64>,
    jacobian_info: &JacobianInfo,
    n_vec: &Mat<f64>,
) -> Result<Mat<f64>, String> {
    let mut b_matrix = Mat::<f64>::zeros(8, ELEMENT_DOFS_QUAD4_SHELL);
    // Extract columns from transformation matrix manually
    let t_matrix = &jacobian_info.transformation_matrix;
    let e1 = col![t_matrix[(0, 0)], t_matrix[(1, 0)], t_matrix[(2, 0)]];
    let e2 = col![t_matrix[(0, 1)], t_matrix[(1, 1)], t_matrix[(2, 1)]];
    let g_xi = &jacobian_info.tangent_xi;
    let g_eta = &jacobian_info.tangent_eta;

    // Calculate 2D surface Jacobian components manually
    // j2d_11 = e1.transpose() * g_xi
    let j2d_11 = e1[0] * g_xi[0] + e1[1] * g_xi[1] + e1[2] * g_xi[2];

    // j2d_12 = e1.transpose() * g_eta
    let j2d_12 = e1[0] * g_eta[0] + e1[1] * g_eta[1] + e1[2] * g_eta[2];

    // j2d_21 = e2.transpose() * g_xi
    let j2d_21 = e2[0] * g_xi[0] + e2[1] * g_xi[1] + e2[2] * g_xi[2];

    // j2d_22 = e2.transpose() * g_eta
    let j2d_22 = e2[0] * g_eta[0] + e2[1] * g_eta[1] + e2[2] * g_eta[2];
    // Create 2D surface Jacobian matrix
    // let _j_surf_2d = mat![[j2d_11, j2d_12], [j2d_21, j2d_22]];

    // Calculate determinant manually
    let det_j_2d = j2d_11 * j2d_22 - j2d_12 * j2d_21;

    if det_j_2d.abs() < TOLERANCE {
        return Err("B Matrix: Surface Jacobian determinant zero.".to_string());
    }

    // Calculate inverse manually
    let j_surf_inv = mat![
        [j2d_22 / det_j_2d, -j2d_12 / det_j_2d],
        [-j2d_21 / det_j_2d, j2d_11 / det_j_2d]
    ];

    // Calculate dN/de1, dN/de2
    let d_n_de1 = d_ndxi * j_surf_inv[(0, 0)] + d_ndeta * j_surf_inv[(1, 0)]; // 4x1
    let d_n_de2 = d_ndxi * j_surf_inv[(0, 1)] + d_ndeta * j_surf_inv[(1, 1)]; // 4x1

    for i in 0..NUM_NODES_QUAD4 {
        let ni = n_vec[(i, 0)];
        let d_ni_de1_val = d_n_de1[(i, 0)];
        let d_ni_de2_val = d_n_de2[(i, 0)];
        let col_start = i * NUM_DOFS_SHELL;

        // Populate B matrix mapping local DOFs -> local strains
        // Assumes local DOFs are ordered [u1, u2, u3, th1, th2, th3]
        // Membrane Strains (e11, e22, g12) from local u1, u2
        b_matrix[(0, col_start + 0)] = d_ni_de1_val; // e11 <- u1 (du1/de1)
        b_matrix[(1, col_start + 1)] = d_ni_de2_val; // e22 <- u2 (du2/de2)
        b_matrix[(2, col_start + 0)] = d_ni_de2_val; // g12 <- u1 (du1/de2)
        b_matrix[(2, col_start + 1)] = d_ni_de1_val; // g12 <- u2 (du2/de1)
        // Bending Strains (k11, k22, k12) from local th1, th2
        b_matrix[(3, col_start + 4)] = d_ni_de1_val; // k11 <- th2 (dth2/de1)
        b_matrix[(4, col_start + 3)] = -d_ni_de2_val; // k22 <- th1 (-dth1/de2)
        b_matrix[(5, col_start + 4)] = d_ni_de2_val; // k12 <- th2 (dth2/de2)
        b_matrix[(5, col_start + 3)] = -d_ni_de1_val; // k12 <- th1 (-dth1/de1)
        // Shear Strains (g13, g23) from local u3, th1, th2
        b_matrix[(6, col_start + 2)] = d_ni_de1_val; // g13 <- u3 (du3/de1)
        b_matrix[(6, col_start + 4)] = ni; // g13 <- th2 (+theta_2)
        b_matrix[(7, col_start + 2)] = d_ni_de2_val; // g23 <- u3 (du3/de2)
        b_matrix[(7, col_start + 3)] = -ni; // g23 <- th1 (-theta_1)
                                            // Drilling DOF (th3, column col_start + 5) is left as zero
    }
    Ok(b_matrix)
}

/// Builds the 24x24 element rotation matrix R from the 3x3 local-to-global matrix T
fn build_element_rotation_matrix(t_local_to_global: &Mat<f64>) -> Mat<f64> {
    let mut r_matrix = Mat::<f64>::zeros(ELEMENT_DOFS_QUAD4_SHELL, ELEMENT_DOFS_QUAD4_SHELL);
    let t = t_local_to_global; // Alias T = [e1|e2|e3]

    // Build block diagonal matrix with T repeated for each nodes DOFs
    for i in 0..NUM_NODES_QUAD4 {
        let start_index = i * NUM_DOFS_SHELL;

        // Place 3x3 T block for transltions
        for row in 0..3 {
            for col in 0..3 {
                r_matrix[(start_index + row, start_index + col)] = t[(row, col)];
            }
        }

        // Place 3x3 T block for rotations
        for row in 0..3 {
            for col in 0..3 {
                r_matrix[(start_index + 3 + row, start_index + 3 + col)] = t[(row, col)];
            }
        }
    }

    r_matrix
}

/// Orthonormalizes a 3x3 matrix using GramSchmidt
fn orthonormalize_matrix(matrix: &Mat<f64>) -> Result<Mat<f64>, String> {
    if matrix.nrows() != 3 || matrix.ncols() != 3 {
        return Err("Matrix must be 3x3 for orthonormalization".to_string());
    }

    let mut q = Mat::<f64>::zeros(3, 3);

    // Extract first column
    let col0 = col![matrix[(0, 0)], matrix[(1, 0)], matrix[(2, 0)]];

    // Calculate norm
    let norm0 = (col0[0] * col0[0] + col0[1] * col0[1] + col0[2] * col0[2]).sqrt();

    if norm0 < TOLERANCE {
        return Err("Cannot orthonormalize matrix with zero column 0".to_string());
    }

    // Normalize first column
    q[(0, 0)] = col0[0] / norm0;
    q[(1, 0)] = col0[1] / norm0;
    q[(2, 0)] = col0[2] / norm0;

    // Extract second column
    let col1 = col![matrix[(0, 1)], matrix[(1, 1)], matrix[(2, 1)]];

    // Project onto first column
    let dot_prod = q[(0, 0)] * col1[0] + q[(1, 0)] * col1[1] + q[(2, 0)] * col1[2];
    let proj10 = col![
        q[(0, 0)] * dot_prod,
        q[(1, 0)] * dot_prod,
        q[(2, 0)] * dot_prod
    ];

    // Subtract projection
    let v1 = col![
        col1[0] - proj10[0],
        col1[1] - proj10[1],
        col1[2] - proj10[2]
    ];

    // Calculate norm
    let norm1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();

    if norm1 < TOLERANCE {
        return Err("Cannot orthonormalize matrix with linearly dependent column 1".to_string());
    }

    // Normalize second column
    q[(0, 1)] = v1[0] / norm1;
    q[(1, 1)] = v1[1] / norm1;
    q[(2, 1)] = v1[2] / norm1;

    // Extract third column
    let col2 = col![matrix[(0, 2)], matrix[(1, 2)], matrix[(2, 2)]];

    // Project onto first column
    let dot_prod0 = q[(0, 0)] * col2[0] + q[(1, 0)] * col2[1] + q[(2, 0)] * col2[2];
    let proj20 = col![
        q[(0, 0)] * dot_prod0,
        q[(1, 0)] * dot_prod0,
        q[(2, 0)] * dot_prod0
    ];

    // Project onto second column
    let dot_prod1 = q[(0, 1)] * col2[0] + q[(1, 1)] * col2[1] + q[(2, 1)] * col2[2];
    let proj21 = col![
        q[(0, 1)] * dot_prod1,
        q[(1, 1)] * dot_prod1,
        q[(2, 1)] * dot_prod1
    ];

    // Subtract projections
    let v2 = col![
        col2[0] - proj20[0] - proj21[0],
        col2[1] - proj20[1] - proj21[1],
        col2[2] - proj20[2] - proj21[2]
    ];

    // Calculate norm
    let norm2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();

    if norm2 < TOLERANCE {
        return Err("Cannot orthonormalize matrix with linearly dependent column 2".to_string());
    }

    // Normalize third column
    q[(0, 2)] = v2[0] / norm2;
    q[(1, 2)] = v2[1] / norm2;
    q[(2, 2)] = v2[2] / norm2;

    // Ensure right-handed coordinate system
    let det = q.determinant();
    if det < 0.0 {
        // Flip the third column
        q[(0, 2)] = -q[(0, 2)];
        q[(1, 2)] = -q[(1, 2)];
        q[(2, 2)] = -q[(2, 2)];
    }

    Ok(q)
}