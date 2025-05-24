use cgmath::Point3;
use log::info;
use noise::{NoiseFn, Perlin};

pub fn generate_terrain(
    width: usize,
    depth: usize,
    scale: f64,
    height_scale: f64,
    perlin_passes: usize,
) -> Vec<Point3<usize>> {
    let t1 = std::time::Instant::now();
    let mut terrain: Vec<Point3<usize>> = Vec::new();
    for seed in 0..perlin_passes {
        let perlin = Perlin::new(seed as u32);
        for x in 0..width {
            for z in 0..depth {
                let idx = x * width + z;
                // Generate noise-based height value
                let noise_value = perlin.get([(x as f64) * scale, (z as f64) * scale]);
                let y = (noise_value * height_scale).round() as usize;

                match terrain.get(idx) {
                    Some(_) => terrain[idx] = Point3::new(x, terrain[idx].y + y, z),
                    None => terrain.push(Point3::new(x, y, z)),
                }
                // Add a point to the terrain
                // terrain.push(Vector3::new(
                //     x as usize,
                //     (height + height / 2.0) as usize,
                //     z as usize,
                // ));
            }
        }
    }
    let terrain = terrain
        .iter()
        .map(|v3| Point3::new(v3.x, v3.y / perlin_passes, v3.z))
        .collect();

    info!("[terrain] (generate_terrain): t={:?})", t1.elapsed());

    terrain
}
