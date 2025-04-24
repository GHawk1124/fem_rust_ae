use crate::constants::{NUM_NODES_QUAD4, PI};
use bevy::log::info;
use faer::{Col, Mat};
use std::collections::HashMap;

// FEM Solver Code (Structs)
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
#[derive(Debug, Clone)]
pub struct Material {
    pub youngs_modulus: f64,
    pub poissons_ratio: f64,
}
#[derive(Debug, Clone)]
pub struct Element {
    pub id: usize,
    pub node_ids: Vec<usize>,
    pub material: Material,
    pub thickness: f64,
}
#[derive(Debug, Clone)]
pub struct FemModel {
    pub nodes: Vec<Node>,
    pub elements: Vec<Element>,
    pub num_dofs_per_node: usize,
    node_id_to_index: HashMap<usize, usize>,
}
#[derive(Debug, Clone)]
pub struct BoundaryCondition {
    pub node_id: usize,
    pub dof_index: usize,
}
#[derive(Debug, Clone)]
pub struct Load {
    pub node_id: usize,
    pub dof_index: usize,
    pub value: f64,
}

/// Holds geometric information calculated at an integration point
#[derive(Debug)]
pub struct JacobianInfo {
    pub tangent_xi: Col<f64>,           // d(x,y,z)/d(xi)
    pub tangent_eta: Col<f64>,          // d(x,y,z)/d(eta)
    pub det_j_surface: f64,             // Surface area scaling factor
    pub transformation_matrix: Mat<f64>, // [T] = [e1 | e2 | e3] (Local coordinate system -> Global coordinates)
}

impl FemModel {
    /// Creates a new FemModel and populates the node ID to index map.
    pub fn new(nodes: Vec<Node>, elements: Vec<Element>, num_dofs_per_node: usize) -> Self {
        let node_id_to_index = nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| (node.id, idx))
            .collect();
        FemModel {
            nodes,
            elements,
            num_dofs_per_node,
            node_id_to_index,
        }
    }
    /// Calculates the total number of degrees of freedom in the model
    #[inline]
    pub fn total_dofs(&self) -> usize {
        self.nodes.len() * self.num_dofs_per_node
    }
    /// Gets the index of a node in the `nodes` vector using its ID
    #[inline]
    pub fn get_node_index(&self, id: usize) -> Option<usize> {
        self.node_id_to_index.get(&id).copied()
    }
    /// Gets the global DOF indices for a given element index
    pub fn get_element_dof_indices(&self, elem_idx: usize) -> Result<Vec<usize>, String> {
        let element = self
            .elements
            .get(elem_idx)
            .ok_or_else(|| format!("Element index {} out of bounds", elem_idx))?;
        if element.node_ids.len() != NUM_NODES_QUAD4 {
            return Err(format!(
                "Element {} is not a Quad4 element (has {} nodes)",
                element.id,
                element.node_ids.len()
            ));
        }
        let mut indices = Vec::with_capacity(crate::constants::ELEMENT_DOFS_QUAD4_SHELL);
        for node_id in &element.node_ids {
            let node_idx = self.get_node_index(*node_id).ok_or(format!(
                "Node ID {} not found for element {}",
                node_id, element.id
            ))?;
            let start_dof = node_idx * self.num_dofs_per_node;
            for i in 0..self.num_dofs_per_node {
                indices.push(start_dof + i);
            }
        }
        Ok(indices)
    }
    /// Gets the global coordinates of the nodes belonging to an element
    pub fn get_element_coords(&self, element: &Element) -> Result<Vec<(f64, f64, f64)>, String> {
        let mut coords = Vec::with_capacity(element.node_ids.len());
        for node_id in &element.node_ids {
            let node_idx = self.get_node_index(*node_id).ok_or_else(|| {
                format!("Node ID {} not found for element {}", node_id, element.id)
            })?;
            let node = &self.nodes[node_idx];
            coords.push((node.x, node.y, node.z));
        }
        Ok(coords)
    }
}

/// Generates FEM mesh data
pub fn generate_fem_mesh(
    radius: f64,
    length: f64,
    thickness: f64,
    nel_l: usize,
    nel_c: usize,
    material: &Material,
    num_dofs_per_node: usize,
) -> Result<FemModel, String> {
    info!("Generating FEM mesh (f64)...");
    let mut nodes = Vec::new();
    let mut elements = Vec::new();
    let mut node_id_counter = 0;
    let dz = length / (nel_l as f64);
    let d_theta = 2.0 * PI / (nel_c as f64);

    for i in 0..=nel_l {
        let z = (i as f64) * dz;
        for j in 0..nel_c {
            let theta = (j as f64) * d_theta;
            let x = radius * theta.cos();
            let y = radius * theta.sin();
            nodes.push(Node {
                id: node_id_counter,
                x,
                y,
                z,
            });
            node_id_counter += 1;
        }
    }

    let mut element_id_counter = 0;
    for i in 0..nel_l {
        for j in 0..nel_c {
            let node_bl_idx = i * nel_c + j;
            let node_br_idx = i * nel_c + (j + 1) % nel_c;
            let node_tr_idx = (i + 1) * nel_c + (j + 1) % nel_c;
            let node_tl_idx = (i + 1) * nel_c + j;
            let node_ids = vec![
                nodes[node_bl_idx].id,
                nodes[node_br_idx].id,
                nodes[node_tr_idx].id,
                nodes[node_tl_idx].id,
            ];
            elements.push(Element {
                id: element_id_counter,
                node_ids,
                material: material.clone(),
                thickness,
            });
            element_id_counter += 1;
        }
    }
    info!(
        "Generated {} FEM nodes and {} FEM elements",
        nodes.len(),
        elements.len()
    );
    Ok(FemModel::new(nodes, elements, num_dofs_per_node))
}