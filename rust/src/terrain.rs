use std::{collections::HashMap, time::Instant};

use block_mesh::ndshape::Shape;
use glam::IVec3;
use noise::{HybridMulti, NoiseFn, Perlin};

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

pub fn generate_chunk_noise<S: Shape<3, Coord = i32>>(
    chunk_wv: &IVec3,
    chunk_shape: &S,
    sampling_rate: usize,
    scale_xz: f64,
    noise: &HybridMulti<Perlin>,
    noise_buffer: &mut [f64],
) -> String {
    let noise_time = Instant::now();
    let sampling_rate_i32 = sampling_rate as i32;
    let chunk_size_i32 = chunk_shape.as_array()[0] as i32;

    let get_base_sample_coord = |w: i32| ((w - 1) / sampling_rate_i32) * sampling_rate_i32 + 1;

    let min_wv_x = chunk_wv[0] - 1;
    let max_wv_x = chunk_wv[0] + chunk_size_i32 - 2;
    let min_wv_z = chunk_wv[2] - 1;
    let max_wv_z = chunk_wv[2] + chunk_size_i32 - 2;

    // Find the min/max sample points we must pre-calculate to cover all lookups.
    let sample_min_x = get_base_sample_coord(min_wv_x);
    let sample_min_z = get_base_sample_coord(min_wv_z);
    // We need to include the next sample point for interpolation at the max boundary.
    let sample_max_x = get_base_sample_coord(max_wv_x) + sampling_rate_i32;
    let sample_max_z = get_base_sample_coord(max_wv_z) + sampling_rate_i32;

    // Create and populate the 2D noise grid (stored in a flat Vec).
    let grid_size_x = ((sample_max_x - sample_min_x) / sampling_rate_i32 + 1) as usize;
    let grid_size_z = ((sample_max_z - sample_min_z) / sampling_rate_i32 + 1) as usize;
    let mut noise_grid = vec![0.0; grid_size_x * grid_size_z];

    for gz in 0..grid_size_z {
        for gx in 0..grid_size_x {
            let current_sample_x = sample_min_x + (gx as i32 * sampling_rate_i32);
            let current_sample_z = sample_min_z + (gz as i32 * sampling_rate_i32);
            let val = noise.get([
                current_sample_x as f64 / scale_xz,
                current_sample_z as f64 / scale_xz,
            ]);
            noise_grid[gz * grid_size_x + gx] = val;
        }
    }

    // This closure performs a fast, direct lookup into our pre-computed grid.
    let get_noise_from_grid = |sample_x: i32, sample_z: i32| {
        let gx = ((sample_x - sample_min_x) / sampling_rate_i32) as usize;
        let gz = ((sample_z - sample_min_z) / sampling_rate_i32) as usize;
        noise_grid[gz * grid_size_x + gx]
    };

    // --- 2. Main Loop with Simplified Bilinear Interpolation ---

    for z in 0..chunk_size_i32 {
        for y in 0..chunk_size_i32 {
            for x in 0..chunk_size_i32 {
                let lv = [x, y, z];
                let wv = [
                    chunk_wv[0] + lv[0] - 1,
                    chunk_wv[1] + lv[1] - 1,
                    chunk_wv[2] + lv[2] - 1,
                ];

                // Calculate the four corners of the sampling cell this point is in.
                let x0_wv = get_base_sample_coord(wv[0]);
                let z0_wv = get_base_sample_coord(wv[2]);
                let x1_wv = x0_wv + sampling_rate_i32;
                let z1_wv = z0_wv + sampling_rate_i32;

                // Calculate interpolation weights (how far the point is between corners).
                let tx = (wv[0] - x0_wv) as f64 / sampling_rate as f64;
                let tz = (wv[2] - z0_wv) as f64 / sampling_rate as f64;

                // Get the 4 corner noise values from our pre-computed grid.
                let c00 = get_noise_from_grid(x0_wv, z0_wv); // Corner at (x0, z0)
                let c10 = get_noise_from_grid(x1_wv, z0_wv); // Corner at (x1, z0)
                let c01 = get_noise_from_grid(x0_wv, z1_wv); // Corner at (x0, z1)
                let c11 = get_noise_from_grid(x1_wv, z1_wv); // Corner at (x1, z1)

                // Perform bilinear interpolation.
                let ix0 = lerp(c00, c10, tx); // Interpolate along X for z0
                let ix1 = lerp(c01, c11, tx); // Interpolate along X for z1
                let noise_value = lerp(ix0, ix1, tz); // Interpolate along Z

                let idx = chunk_shape.linearize(lv) as usize;
                noise_buffer[idx] = noise_value;
            }
        }
    }

    format!("[thread] (noise_time) - t={:?}", noise_time.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_mesh::ndshape::{ConstShape, ConstShape3i32};
    type ChunkShape4 = ConstShape3i32<4, 4, 4>;
    type ChunkShape10 = ConstShape3i32<10, 10, 10>;

    // fn tri_lerp_origin_points() {
    //     let sampling_rate = 2;
    //     let chunk_shape = ChunkShape4 {};
    //     let mut noise_buffer = [f64::NAN; ChunkShape4::SIZE as usize];

    //     noise_buffer[chunk_shape.linearize([1, 1, 1]) as usize] = 0.0; //c000
    //     noise_buffer[chunk_shape.linearize([3, 1, 1]) as usize] = 1.0; //c100
    //     noise_buffer[chunk_shape.linearize([1, 3, 1]) as usize] = 2.0; //c010
    //     noise_buffer[chunk_shape.linearize([1, 1, 3]) as usize] = 4.0; //c001
    //     noise_buffer[chunk_shape.linearize([3, 3, 1]) as usize] = 3.0; //c110
    //     noise_buffer[chunk_shape.linearize([3, 1, 3]) as usize] = 5.0; //c101
    //     noise_buffer[chunk_shape.linearize([1, 3, 3]) as usize] = 6.0; //c011
    //     noise_buffer[chunk_shape.linearize([3, 3, 3]) as usize] = 7.0; //c111

    //     let x_axis = tri_lerp(
    //         chunk_shape.linearize([2, 1, 1]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(x_axis, 0.5);

    //     let y_axis = tri_lerp(
    //         chunk_shape.linearize([1, 2, 1]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(y_axis, 1.0);

    //     let z_axis = tri_lerp(
    //         chunk_shape.linearize([1, 1, 2]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(z_axis, 2.0);

    //     let diag = tri_lerp(
    //         chunk_shape.linearize([2, 2, 2]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(diag, 3.5);
    // }

    // #[test]
    // fn tri_lerp_offset_points() {
    //     let sampling_rate = 4;
    //     let chunk_shape = ChunkShape10 {};
    //     let mut noise_buffer = [f64::NAN; ChunkShape10::SIZE as usize];

    //     noise_buffer[chunk_shape.linearize([5, 5, 5]) as usize] = 0.0; //c000
    //     noise_buffer[chunk_shape.linearize([9, 5, 5]) as usize] = 4.0; //c100
    //     noise_buffer[chunk_shape.linearize([5, 9, 5]) as usize] = 8.0; //c010
    //     noise_buffer[chunk_shape.linearize([5, 5, 9]) as usize] = 16.0; //c001
    //     noise_buffer[chunk_shape.linearize([9, 9, 5]) as usize] = 12.0; //c110
    //     noise_buffer[chunk_shape.linearize([9, 5, 9]) as usize] = 20.0; //c101
    //     noise_buffer[chunk_shape.linearize([5, 9, 9]) as usize] = 24.0; //c011
    //     noise_buffer[chunk_shape.linearize([9, 9, 9]) as usize] = 28.0; //c111

    //     let x_axis = tri_lerp(
    //         chunk_shape.linearize([6, 5, 5]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(x_axis, 1.0);

    //     let y_axis = tri_lerp(
    //         chunk_shape.linearize([5, 6, 5]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(y_axis, 2.0);

    //     let z_axis = tri_lerp(
    //         chunk_shape.linearize([5, 5, 6]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(z_axis, 4.0);

    //     let diag_1 = tri_lerp(
    //         chunk_shape.linearize([6, 6, 6]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(diag_1, 7.0);

    //     let diag_2 = tri_lerp(
    //         chunk_shape.linearize([7, 7, 7]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(diag_2, 14.0);

    //     let diag_3 = tri_lerp(
    //         chunk_shape.linearize([8, 8, 8]) as usize,
    //         &chunk_shape,
    //         sampling_rate,
    //         &mut noise_buffer,
    //     );
    //     assert_eq!(diag_3, 21.0);
    // }
    #[test]
    fn generate_chunk_noise() {
        let chunk_wv = IVec3::new(0, 0, 0);
        let chunk_shape = ChunkShape4 {};
        let sampling_rate = 2;
        let mut noise = HybridMulti::<Perlin>::new(1234);
        noise.octaves = 5;
        noise.frequency = 1.1;
        noise.lacunarity = 2.8;
        noise.persistence = 0.4;
        let mut noise_buffer = [f64::NAN; ChunkShape4::SIZE as usize];

        // super::generate_chunk_noise(
        //     &chunk_wv,
        //     &chunk_shape,
        //     sampling_rate,
        //     1.0,
        //     &noise,
        //     &mut noise_buffer,
        // );

        // let [x, y, z] = [1, 1, 1];
        // let c000 = noise_buffer[chunk_shape.linearize([x, y, z]) as usize];
        // assert_eq!(c000, noise.get([(x - 1) as f64, (z - 1) as f64]));

        // let real_value_points = [
        //     [1, 1, 1],
        //     [1, 1, 3],
        //     [1, 3, 1],
        //     [1, 3, 3],
        //     [3, 1, 1],
        //     [3, 1, 3],
        //     [3, 3, 1],
        //     [3, 3, 3],
        // ];

        // for [x, y, z] in real_value_points {
        //     let expected = noise.get([(x - 1) as f64, (z - 1) as f64]);
        //     let actual = noise_buffer[chunk_shape.linearize([x, y, z]) as usize];
        //     assert_eq!(expected, actual);
        // }

        // let interpolated_value_points = [
        //     // [1, 1, 1], origin => noise value
        //     [1, 1, 2],
        //     [1, 2, 1],
        //     [1, 2, 2],
        //     [2, 1, 1],
        //     [2, 1, 2],
        //     [2, 2, 1],
        //     [2, 2, 2],
        // ];

        let chunk_shape = ChunkShape4 {};
        let sampling_rate = 2;
        let scale_xy = 1000.0;

        let chunk_v_10_13_1 = IVec3::new(10, 13, 1);
        let chunk_wv_10_13_1 = chunk_v_10_13_1 * 2;
        let mut noise_buffer_10_13_1 = [f64::NAN; ChunkShape4::SIZE as usize];
        super::generate_chunk_noise(
            &chunk_wv_10_13_1,
            &chunk_shape,
            sampling_rate,
            scale_xy,
            &noise,
            &mut noise_buffer_10_13_1,
        );

        let chunk_v_10_13_2 = IVec3::new(10, 13, 2);
        let chunk_wv_10_13_2 = chunk_v_10_13_2 * 2;
        let mut noise_buffer_10_13_2 = [f64::NAN; ChunkShape4::SIZE as usize];
        super::generate_chunk_noise(
            &chunk_wv_10_13_2,
            &chunk_shape,
            sampling_rate,
            scale_xy,
            &noise,
            &mut noise_buffer_10_13_2,
        );

        // println!("noise_buffer_10_13_1={noise_buffer_10_13_1:?}");
        // println!("noise_buffer_10_13_2={noise_buffer_10_13_2:?}");
        assert_eq!(true, false)

        // for [x, y, z] in interpolated_value_points {
        //     let idx = chunk_shape.linearize([x, y, z]) as usize;
        //     let expected = tri_lerp(idx, &chunk_shape, sampling_rate, &mut noise_buffer);
        //     let actual = noise_buffer[idx];
        //     assert_eq!(expected, actual);
        // }
    }
}
