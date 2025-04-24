use crate::{
    constants::*,
    fem_model::{generate_fem_mesh, BoundaryCondition, Load, Material, Node},
    fem_solver::StaticSolver,
};
use bevy::log::{error, info};
use plotters::prelude::*;
use plotters::style::Color;

// Struct to hold convergence study results
#[derive(Debug, Clone)]
pub struct ConvergenceStudyResults {
    pub element_counts: Vec<usize>,
    pub max_displacements: Vec<f32>,
}

/// Run a convergence study and generate a plot showing max axial displacement vs. number of elements
#[cfg(not(target_arch = "wasm32"))]
pub fn run_convergence_study() -> Result<ConvergenceStudyResults, Box<dyn std::error::Error>> {
    info!("Starting Convergence Study");

    let material = Material {
        youngs_modulus: YOUNG_MODULUS,
        poissons_ratio: POISSONS_RATIO,
    };

    // Define mesh densities to test
    let axial_element_counts = [5, 10, 15, 20, 25];
    let circumferential_element_count = 30; // Keep this constant

    let mut element_counts = Vec::new();
    let mut max_displacements = Vec::new();

    for &axial_count in &axial_element_counts {
        let total_elements = axial_count * circumferential_element_count;
        info!(
            "Running convergence study with {} axial elements ({} total elements)",
            axial_count, total_elements
        );

        // Generate mesh with current density (using function from fem_model)
        let fem_model = match generate_fem_mesh(
            CYLINDER_RADIUS_FEM,
            CYLINDER_LENGTH_FEM,
            CYLINDER_THICKNESS_FEM,
            axial_count,
            circumferential_element_count,
            &material,
            NUM_DOFS_SHELL,
        ) {
            Ok(model) => model,
            Err(e) => {
                error!("Failed to generate FEM mesh for convergence study: {}", e);
                continue;
            }
        };

        // Create boundary conditions (fixed at z=0)
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

        // Create loads (axial force at z=length)
        let mut loads = Vec::new();
        let axial_dof_index = 2;
        let top_nodes: Vec<&Node> = fem_model // Specify Node type from fem_model
            .nodes
            .iter()
            .filter(|n| (n.z - CYLINDER_LENGTH_FEM).abs() < TOLERANCE)
            .collect();

        if top_nodes.is_empty() {
            error!("No nodes found at the top edge to apply load");
            continue;
        }

        let load_per_node = TOTAL_APPLIED_FORCE / (top_nodes.len() as f64);
        for node in &top_nodes {
            loads.push(Load {
                node_id: node.id,
                dof_index: axial_dof_index,
                value: load_per_node,
            });
        }

        // Solve the FEM problem
        let solver = StaticSolver::new(&fem_model, &bcs, &loads);
        match solver.solve() {
            Ok(displacement_vector) => {
                // Find maximum absolute axial displacement
                let mut max_disp = 0.0f32;
                for node_idx in 0..fem_model.nodes.len() {
                    let disp_z = displacement_vector[node_idx * NUM_DOFS_SHELL + 2] as f32;
                    max_disp = max_disp.max(disp_z.abs()); // Use abs for convergence plot
                }

                info!(
                    "Convergence Study: Elements = {}, Max Z Displacement = {:.4e}",
                    total_elements, max_disp
                );

                // Store results
                element_counts.push(total_elements);
                max_displacements.push(max_disp); // Use absolute value for plotting
            }
            Err(e) => {
                error!("FEM solver failed for convergence study: {}", e);
                continue;
            }
        }
    }

    info!(
        "Convergence Study Complete: {} data points",
        element_counts.len()
    );

    // Create the results struct
    let results = ConvergenceStudyResults {
        element_counts,
        max_displacements,
    };

    // Generate the plot
    create_convergence_plot(&results)?;

    Ok(results)
}

/// WASM stub for convergence study
#[cfg(target_arch = "wasm32")]
pub fn run_convergence_study() -> Result<ConvergenceStudyResults, Box<dyn std::error::Error>> {
    info!("Convergence study is not supported in WASM");
    Err("Convergence study is not supported in WASM".into())
}

/// Create a plot of the convergence study results using the plotters crate
#[cfg(not(target_arch = "wasm32"))]
fn create_convergence_plot(
    results: &ConvergenceStudyResults,
) -> Result<(), Box<dyn std::error::Error>> {
    if results.element_counts.is_empty() || results.max_displacements.is_empty() {
        return Err("No data points to plot".into());
    }

    info!("Creating convergence study plots");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // Create plots in both PNG and SVG formats
    create_plot_with_backend(
        results,
        BitMapBackend::new("output/convergence_study.png", (800, 600)),
    )?;
    create_plot_with_backend(
        results,
        SVGBackend::new("output/convergence_study.svg", (800, 600)),
    )?;

    info!(
        "Convergence study plots saved to output/convergence_study.png and output/convergence_study.svg"
    );

    Ok(())
}

/// Helper function to create a plot with a specific backend
#[cfg(not(target_arch = "wasm32"))]
fn create_plot_with_backend<DB: DrawingBackend>(
    results: &ConvergenceStudyResults,
    backend: DB,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static + std::error::Error, // Added Error bound for present()
{
    let root = backend.into_drawing_area();

    // Fill the background with white
    root.fill(&WHITE)?;

    // Find min/max values for axes
    let min_elements = *results.element_counts.iter().min().unwrap_or(&0) as f32;
    let max_elements = *results.element_counts.iter().max().unwrap_or(&1000) as f32;
    let min_disp = results
        .max_displacements
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(0.0);
    let max_disp = results
        .max_displacements
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(1.0);

    // Add some padding to the bounds
    let x_range = (max_elements - min_elements).max(1.0); // Avoid zero range
    let y_range = (max_disp - min_disp).max(1e-9);   // Avoid zero range
    let x_padding = x_range * 0.1;
    let y_padding = y_range * 0.1;

    let x_min = (min_elements - x_padding).floor(); // Use floor/ceil for better bounds
    let x_max = (max_elements + x_padding).ceil();
    let y_min = min_disp - y_padding;
    let y_max = max_disp + y_padding;


    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Convergence Study: Max Abs Axial Disp vs. Number of Elements", // Updated title
            ("sans-serif", 22).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?; // Use calculated ranges

    // Configure the chart
    chart
        .configure_mesh()
        .x_desc("Number of Elements")
        .y_desc("Max Absolute Axial Displacement (m)") // Updated axis label
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Create data series
    let data_points: Vec<(f32, f32)> = results
        .element_counts
        .iter()
        .zip(results.max_displacements.iter())
        .map(|(&elements, &displacement)| (elements as f32, displacement))
        .collect();

    // Draw the line series
    chart
        .draw_series(LineSeries::new(data_points.clone(), &RED))?
        .label("Max Displacement")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3))); // Use RED directly

    // Draw the point series
    chart.draw_series(PointSeries::of_element(
        data_points,
        5,
        RED.filled(), // Use RED directly
        &|c, s, st| { // Use filled circle
            EmptyElement::at(c) // Position
            + Circle::new((0,0),s,st) // Circle shape
        },
    ))?;

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8)) // Use WHITE directly
        .border_style(BLACK) // Use BLACK directly
        .draw()?;

    // Save the plot
    root.present().map_err(|e| e.into()) // Convert backend error to Box<dyn Error>

}