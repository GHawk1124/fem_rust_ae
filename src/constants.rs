pub use std::f64::consts::PI;

// Configuration
pub const CYLINDER_RADIUS_FEM: f64 = 4.5;
pub const CYLINDER_LENGTH_FEM: f64 = 4.0;
pub const CYLINDER_THICKNESS_FEM: f64 = 0.008; // 8mm
pub const YOUNG_MODULUS: f64 = 200e9; // Steel (Pa)
pub const POISSONS_RATIO: f64 = 0.3;
pub const NUM_ELEMENTS_AXIAL_FEM: usize = 15; // Mesh density along length
pub const NUM_ELEMENTS_CIRCUMFERENTIAL_FEM: usize = 30; // Mesh density around circumference
pub const NUM_DOFS_SHELL: usize = 6; // ux, uy, uz, rx, ry, rz
pub const TOTAL_APPLIED_FORCE: f64 = -1.0e7; // Apply a larger force (-10 MN)

// Visualization Configuration
pub const VIS_DEFORMATION_SCALE: f32 = 50.0; // Scale factor for visualizing displacements
pub const NUM_NODES_QUAD4: usize = 4;
pub const ELEMENT_DOFS_QUAD4_SHELL: usize = NUM_NODES_QUAD4 * NUM_DOFS_SHELL; // 24
pub const TOLERANCE: f64 = 1e-12; // Small number for floating point comparisons