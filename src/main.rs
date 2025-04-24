use bevy::{
    log::{error, info},
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::PrimitiveTopology,
    },
    window::{PresentMode, Window},
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use faer::Col;

mod constants;
mod fem_model;
mod fem_solver;
mod convergence;

use constants::*;
use fem_model::{generate_fem_mesh, BoundaryCondition, FemModel, Load, Material, Node};
use fem_solver::StaticSolver;

// Bevy Application Code
#[derive(Debug, Clone)]
struct VisNode {
    #[allow(dead_code)]
    id: usize,
    position: Vec3,
    normal: Vec3,
    displacement_z: f32,
}
#[derive(Debug, Clone)]
struct VisElement {
    #[allow(dead_code)]
    id: usize,
    node_indices: [usize; 4],
}
#[derive(Resource)]
struct FemResult {
    displacement: Col<f64>,
    fem_model: FemModel,
    min_disp_z: f32,
    max_disp_z: f32,
}

/// Resource to store the simulation configuration parameters
#[derive(Resource)]
struct SimulationConfig {
    // Geometry parameters
    radius: f64,
    length: f64,
    thickness: f64,

    // Material parameters
    youngs_modulus: f64,
    poissons_ratio: f64,

    // Mesh parameters
    elements_axial: usize,
    elements_circumferential: usize,

    // Load parameters
    total_force: f64,

    // Visualization parameters
    deformation_scale: f32,

    // Visualization options
    show_original_mesh: bool,
    show_deformed_mesh: bool,

    // Simulation control
    run_simulation: bool,
    has_results: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            radius: CYLINDER_RADIUS_FEM,
            length: CYLINDER_LENGTH_FEM,
            thickness: CYLINDER_THICKNESS_FEM,
            youngs_modulus: YOUNG_MODULUS,
            poissons_ratio: POISSONS_RATIO,
            elements_axial: NUM_ELEMENTS_AXIAL_FEM,
            elements_circumferential: NUM_ELEMENTS_CIRCUMFERENTIAL_FEM,
            total_force: TOTAL_APPLIED_FORCE,
            deformation_scale: VIS_DEFORMATION_SCALE,
            show_original_mesh: true,
            show_deformed_mesh: true,
            run_simulation: false,
            has_results: false,
        }
    }
}

fn main() {
    // Check if we should run a convergence study
    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.len() > 1 && args[1] == "convergence" {
            info!("Running convergence study mode");
            match convergence::run_convergence_study() {
                Ok(_results) => {
                    info!("Convergence study completed successfully");
                    info!("Results saved to output/convergence_study.png");
                    std::process::exit(0);
                }
                Err(e) => {
                    error!("Convergence study failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    // Run normal visualization mode
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy FEM Static Viewer".into(),
                present_mode: PresentMode::AutoVsync,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .init_resource::<SimulationConfig>()
        .add_systems(Startup, setup_camera)
        // Use a different approach for the UI system
        .add_systems(Update, ui_system)
        .add_systems(Update, (run_fem_solver, setup_visualization).chain())
        .run();
}

/// Runs the FEM static solver and inserts results as a Bevy resource
fn run_fem_solver(mut commands: Commands, mut sim_config: ResMut<SimulationConfig>) {
    // Only run the simulation if the run_simulation flag is set
    if !sim_config.run_simulation {
        return;
    }

    // Reset the flags
    sim_config.run_simulation = false;
    sim_config.has_results = false;

    info!("Starting FEM Static Analysis");
    let material = Material {
        youngs_modulus: sim_config.youngs_modulus,
        poissons_ratio: sim_config.poissons_ratio,
    };

    // Pass arguments to generate_fem_mesh (now in fem_model module)
    let fem_model = match generate_fem_mesh(
        sim_config.radius,
        sim_config.length,
        sim_config.thickness,
        sim_config.elements_axial,
        sim_config.elements_circumferential,
        &material,
        NUM_DOFS_SHELL,
    ) {
        Ok(model) => model,
        Err(e) => {
            error!("FATAL: Failed to generate FEM mesh: {}", e);
            std::process::exit(1);
        }
    };

    let mut bcs = Vec::new();
    for node in fem_model.nodes.iter() {
        if node.z.abs() < TOLERANCE {
            for dof_idx in 0..NUM_DOFS_SHELL {
                bcs.push(BoundaryCondition {
                    node_id: node.id,
                    dof_index: dof_idx,
                });
            }
        }
    }
    info!("FEM Setup: Defined {} BC constraints", bcs.len());

    let mut loads = Vec::new();
    let axial_dof_index = 2;
    let top_nodes: Vec<&Node> = fem_model
        .nodes
        .iter()
        .filter(|n| (n.z - sim_config.length).abs() < TOLERANCE)
        .collect();
    if top_nodes.is_empty() {
        error!("FATAL: No nodes found at the top edge to apply load");
        std::process::exit(1);
    }
    let load_per_node = sim_config.total_force / (top_nodes.len() as f64);
    for node in &top_nodes {
        loads.push(Load {
            node_id: node.id,
            dof_index: axial_dof_index,
            value: load_per_node,
        });
    }
    info!("FEM Setup: Defined {} nodal loads", loads.len());

    let solver = StaticSolver::new(&fem_model, &bcs, &loads);

    match solver.solve() {
        Ok(displacement_vector) => {
            info!("FEM Static Analysis Complete");
            let mut min_disp = f32::MAX;
            let mut max_disp = f32::MIN;
            for node_idx in 0..fem_model.nodes.len() {
                let disp_z = displacement_vector[node_idx * NUM_DOFS_SHELL + 2] as f32;
                min_disp = min_disp.min(disp_z);
                max_disp = max_disp.max(disp_z);
            }
            info!("FEM Result: Min Z Displacement = {:.4e}", min_disp);
            info!("FEM Result: Max Z Displacement = {:.4e}", max_disp);
            commands.insert_resource(FemResult {
                displacement: displacement_vector,
                fem_model, // fem_model is moved here
                min_disp_z: min_disp,
                max_disp_z: max_disp,
            });
        }
        Err(e) => {
            error!("FATAL: FEM solver failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// Sets up the camera for the 3D view
fn setup_camera(mut commands: Commands) {
    // Add a camera positioned to see the cylinder
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(-10.0, 8.0, 10.0).looking_at(
                Vec3::new(0.0, 0.0, CYLINDER_LENGTH_FEM as f32 / 2.0),
                Vec3::Y,
            ),
            ..Default::default()
        })
        .insert(PanOrbitCamera {
            focus: Vec3::new(0.0, 0.0, CYLINDER_LENGTH_FEM as f32 / 2.0),
            radius: Some(15.0),
            ..Default::default()
        });

    // Add a light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 2_000_000.0,
            shadows_enabled: true,
            range: 100.0,
            ..Default::default()
        },
        transform: Transform::from_xyz(8.0, 10.0, 8.0),
        ..Default::default()
    });
}

/// Sets up Bevy visualization using static FEM results
fn setup_visualization(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    fem_result: Option<Res<FemResult>>,
    mut sim_config: ResMut<SimulationConfig>,
    mesh_entities: Query<Entity, With<Handle<Mesh>>>,
) {
    // Only run if we have results and they haven't been visualized yet
    if fem_result.is_none() || sim_config.has_results {
        return;
    }

    // Mark that we have visualized the results
    sim_config.has_results = true;

    // Remove any existing meshes
    for entity in mesh_entities.iter() {
        commands.entity(entity).despawn();
    }

    let fem_result = fem_result.unwrap();
    info!("Setting up Bevy Visualization");
    let fem_model = &fem_result.fem_model;
    let displacement = &fem_result.displacement;
    let min_disp = fem_result.min_disp_z;
    let max_disp = fem_result.max_disp_z;
    // Displacement range is handled in create_mesh function

    let (original_vis_nodes, deformed_vis_nodes, vis_elements) =
        generate_visualization_mesh_data(fem_model, displacement, sim_config.deformation_scale);

    // Create original mesh (gray color)
    let original_mesh = create_mesh(&original_vis_nodes, &vis_elements, None);

    // Create deformed mesh (colored by displacement)
    let deformed_mesh = create_mesh(
        &deformed_vis_nodes,
        &vis_elements,
        Some((min_disp, max_disp)),
    );

    // Spawn original mesh (semi-transparent)
    if sim_config.show_original_mesh {
        commands.spawn(PbrBundle {
            mesh: meshes.add(original_mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.7, 0.7, 0.7).with_alpha(0.3),
                alpha_mode: AlphaMode::Blend,
                perceptual_roughness: 0.8,
                metallic: 0.1,
                cull_mode: None, // Render both sides
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..Default::default()
        });
    }

    // Spawn deformed mesh in same location
    if sim_config.show_deformed_mesh {
        commands.spawn(PbrBundle {
            mesh: meshes.add(deformed_mesh),
            material: materials.add(StandardMaterial {
                perceptual_roughness: 0.8,
                metallic: 0.1,
                cull_mode: None, // Render both sides
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..Default::default()
        });
    }

    info!("Bevy Visualization Setup Complete");
}

/// Helper function to create a mesh from nodes and elements
fn create_mesh(
    vis_nodes: &[VisNode],
    vis_elements: &[VisElement],
    disp_range: Option<(f32, f32)>, // (min_disp_z, max_disp_z)
) -> Mesh {
    let num_triangles = vis_elements.len() * 2;
    let num_vertices = num_triangles * 3;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(num_vertices);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(num_vertices);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(num_vertices);
    let mut indices: Vec<u32> = Vec::with_capacity(num_vertices);

    // Define colors for displacement visualization
    let color_min = Color::srgb(0.0, 0.0, 1.0).to_linear();
    let color_max = Color::srgb(1.0, 0.0, 0.0).to_linear();

    // Default color (gray) if no displacement range is provided
    let default_color = [0.7, 0.7, 0.7, 1.0];

    for element in vis_elements {
        let idx0 = element.node_indices[0];
        let idx1 = element.node_indices[1];
        let idx2 = element.node_indices[2];
        let idx3 = element.node_indices[3];
        let n0 = &vis_nodes[idx0];
        let n1 = &vis_nodes[idx1];
        let n2 = &vis_nodes[idx2];
        let n3 = &vis_nodes[idx3];

        // Calculate colors based on displacement if range is provided
        let (color1_array, color2_array) = if let Some((min_disp, max_disp)) = disp_range {
            let disp_range = (max_disp - min_disp).max(1e-9); // Avoid division by zero

            // Use individual node displacements for smoother color transition per vertex
            let calc_color = |node: &VisNode| -> [f32; 4] {
                let t = ((node.displacement_z - min_disp) / disp_range).clamp(0.0, 1.0);
                [
                    color_min.red * (1.0 - t) + color_max.red * t,
                    color_min.green * (1.0 - t) + color_max.green * t,
                    color_min.blue * (1.0 - t) + color_max.blue * t,
                    1.0,
                ]
            };

            let c0 = calc_color(n0);
            let c1 = calc_color(n1);
            let c2 = calc_color(n2);
            let c3 = calc_color(n3);

             // First triangle colors (n0, n1, n2)
             let color1_tri = [c0, c1, c2];
             // Second triangle colors (n0, n2, n3)
             let color2_tri = [c0, c2, c3];
             (color1_tri, color2_tri)

        } else {
             // If no displacement range, use default gray for all vertices
            ([default_color; 3], [default_color; 3])
        };


        // First triangle (0, 1, 2)
        let start_index = positions.len() as u32;
        positions.push(n0.position.into());
        positions.push(n1.position.into());
        positions.push(n2.position.into());
        normals.push(n0.normal.into());
        normals.push(n1.normal.into());
        normals.push(n2.normal.into());
        if disp_range.is_some() {
            colors.extend_from_slice(&color1_array);
        } else {
            colors.extend_from_slice(&[default_color, default_color, default_color]);
        }
        indices.extend_from_slice(&[start_index, start_index + 1, start_index + 2]);


        // Second triangle (0, 2, 3)
        let start_index = positions.len() as u32;
        positions.push(n0.position.into());
        positions.push(n2.position.into());
        positions.push(n3.position.into());
        normals.push(n0.normal.into());
        normals.push(n2.normal.into());
        normals.push(n3.normal.into());
         if disp_range.is_some() {
            colors.extend_from_slice(&color2_array);
        } else {
            colors.extend_from_slice(&[default_color, default_color, default_color]);
        }
        indices.extend_from_slice(&[start_index, start_index + 1, start_index + 2]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        VertexAttributeValues::Float32x4(colors),
    );
    mesh.insert_indices(Indices::U32(indices));

    mesh
}


/// UI system that displays the egui controls
fn ui_system(
    mut contexts: EguiContexts,
    mut sim_config: ResMut<SimulationConfig>,
    fem_result: Option<Res<FemResult>>,
) {
    // Get the context for the primary window, if it exists
    let ctx = match contexts.try_ctx_mut() {
        Some(ctx) => ctx,
        None => {
            // Skip this frame if the context isn't ready
            return;
        }
    };

    egui::SidePanel::right("settings_panel")
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.heading("FEM Simulation Settings");
            ui.add_space(10.0);

            // Track changes to parameters that require re-running visualization
            let mut vis_params_changed = false;

            ui.collapsing("Geometry", |ui| {
                vis_params_changed |= ui.add(egui::Slider::new(&mut sim_config.radius, 1.0..=10.0).text("Radius (m)")).changed();
                vis_params_changed |= ui.add(egui::Slider::new(&mut sim_config.length, 1.0..=10.0).text("Length (m)")).changed();
                vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.thickness, 0.001..=0.05)
                        .text("Thickness (m)"),
                ).changed();
            });

            ui.collapsing("Material", |ui| {
                vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.youngs_modulus, 50e9..=300e9)
                        .text("Young's Modulus (Pa)")
                        .logarithmic(true),
                ).changed();
                vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.poissons_ratio, 0.0..=0.49)
                        .text("Poisson's Ratio"),
                ).changed();
            });

            ui.collapsing("Mesh", |ui| {
                vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.elements_axial, 5..=30)
                        .text("Axial Elements"),
                ).changed();
                 vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.elements_circumferential, 10..=60)
                        .text("Circumferential Elements"),
                ).changed();
            });

             ui.collapsing("Load", |ui| {
                 vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.total_force, -1.0e8..=-1.0e6)
                        .text("Total Force (N)")
                        .logarithmic(true), // Consider if log makes sense for negative
                ).changed();
            });


            ui.collapsing("Visualization", |ui| {
                 // Changes here only affect visualization, not the simulation results
                vis_params_changed |= ui.add(
                    egui::Slider::new(&mut sim_config.deformation_scale, 1.0..=200.0)
                        .text("Deformation Scale"),
                ).changed();
                 vis_params_changed |= ui.checkbox(&mut sim_config.show_original_mesh, "Show Original Mesh").changed();
                 vis_params_changed |= ui.checkbox(&mut sim_config.show_deformed_mesh, "Show Deformed Mesh").changed();

            });

            ui.add_space(20.0);

             // If visualization parameters changed and we have results, mark results as outdated
             // so visualization will update without re-running the whole simulation.
            if vis_params_changed && fem_result.is_some() {
                sim_config.has_results = false; // Trigger re-visualization
                info!("Visualization parameters changed - scheduling visualization update");
            }


            if ui.button("Run Simulation").clicked() {
                sim_config.run_simulation = true;
                sim_config.has_results = false; // Reset results flag
                info!("Run button clicked - Starting simulation");
            }


            if let Some(result) = fem_result {
                // Only display results info if the results are considered current
                if sim_config.has_results {
                     ui.add_space(10.0);
                    ui.heading("Results");
                    ui.label(format!("Min Z Displacement: {:.4e} m", result.min_disp_z));
                    ui.label(format!("Max Z Displacement: {:.4e} m", result.max_disp_z));
                    ui.label(format!(
                        "Total Elements: {}",
                        result.fem_model.elements.len()
                    ));
                    ui.label(format!("Total Nodes: {}", result.fem_model.nodes.len()));
                    ui.label(format!("Total DOFs: {}", result.fem_model.total_dofs()));
                }
            }
        });
}

/// Generates visualiation mesh data for both original and deformed shapes
fn generate_visualization_mesh_data(
    fem_model: &FemModel,
    displacement: &Col<f64>, // Global displacement vector
    deformation_scale: f32,
) -> (Vec<VisNode>, Vec<VisNode>, Vec<VisElement>) {
    info!("Generating visualization mesh data (original and deformed)...");
    let mut original_vis_nodes = Vec::with_capacity(fem_model.nodes.len());
    let mut deformed_vis_nodes = Vec::with_capacity(fem_model.nodes.len());

    for (node_idx, fem_node) in fem_model.nodes.iter().enumerate() {
        let start_dof = node_idx * fem_model.num_dofs_per_node;

        // Extract displacements using indexing
        let dx = displacement[start_dof + 0];
        let dy = displacement[start_dof + 1];
        let dz = displacement[start_dof + 2];

        // Calculate positions
        let original_pos = Vec3::new(fem_node.x as f32, fem_node.y as f32, fem_node.z as f32);
        let displacement_vec = Vec3::new(dx as f32, dy as f32, dz as f32);
        let deformed_pos = original_pos + displacement_vec * deformation_scale;

        // Calculate normal based on original geometry (radial direction for cylinder)
        let normal = Vec3::new(fem_node.x as f32, fem_node.y as f32, 0.0).normalize_or_zero();

        original_vis_nodes.push(VisNode {
            id: fem_node.id,
            position: original_pos,
            normal,
            displacement_z: 0.0, // Original mesh has zero displacement for coloring
        });

        deformed_vis_nodes.push(VisNode {
            id: fem_node.id,
            position: deformed_pos,
            normal, // Use original normal for consistent shading
            displacement_z: dz as f32, // Store actual displacement for coloring
        });
    }

    // Create visualization elements
    let mut vis_elements = Vec::with_capacity(fem_model.elements.len());
    for fem_element in &fem_model.elements {
        // Find the corresponding indices in the vis_nodes list using the model's map
        let node_bl_vis_idx = fem_model.get_node_index(fem_element.node_ids[0]).unwrap();
        let node_br_vis_idx = fem_model.get_node_index(fem_element.node_ids[1]).unwrap();
        let node_tr_vis_idx = fem_model.get_node_index(fem_element.node_ids[2]).unwrap();
        let node_tl_vis_idx = fem_model.get_node_index(fem_element.node_ids[3]).unwrap();

        vis_elements.push(VisElement {
            id: fem_element.id,
            node_indices: [
                node_bl_vis_idx,
                node_br_vis_idx,
                node_tr_vis_idx,
                node_tl_vis_idx,
            ],
        });
    }

    (original_vis_nodes, deformed_vis_nodes, vis_elements)
}