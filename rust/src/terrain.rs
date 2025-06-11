use std::collections::HashMap;

use crate::voxel::Block;
use block_mesh::ndshape::Shape;
use glam::IVec3;
use noise::{HybridMulti, NoiseFn, Perlin, Vector3};

// Assumes a chunk with a 1 layer padding
// Given a world chunk of (32,32,32), as 34x34x34 shaped should be passed in
// (1,1,1) in this padded chunk will be equivalent to (0,0,0) in normal chunk
pub fn generate_chunk_noise<S: Shape<3, Coord = i32>>(
    chunk_wv: &IVec3,
    chunk_shape: &S,
    sampling_rate: usize,
    scale_xz: f64,
    noise: &HybridMulti<Perlin>,
    noise_buffer: &mut [f64],
) {
    generate(
        chunk_wv,
        chunk_shape,
        sampling_rate,
        scale_xz,
        noise,
        noise_buffer,
    );
    interpolate(chunk_shape, sampling_rate, noise_buffer);
}

fn generate<S: Shape<3, Coord = i32>>(
    chunk_wv: &IVec3,
    chunk_shape: &S,
    sampling_rate: usize,
    scale_xz: f64,
    noise: &HybridMulti<Perlin>,
    noise_buffer: &mut [f64],
) {
    let chunk_size_i = chunk_shape.as_array()[0] as i32;
    for z in (1..chunk_size_i).step_by(sampling_rate) {
        for y in (1..chunk_size_i).step_by(sampling_rate) {
            for x in (1..chunk_size_i).step_by(sampling_rate) {
                // should this be linearized? Don't think wv will track to noise value unless the same conversions are made
                // might not matter if consistent
                let idx = chunk_shape.linearize([x, y, z]) as usize;
                noise_buffer[idx] = noise.get([
                    ((chunk_wv.x + x - 1) as f64) / scale_xz,
                    ((chunk_wv.z + z - 1) as f64 / scale_xz),
                ]);
            }
        }
    }
}

fn interpolate<S: Shape<3, Coord = i32>>(
    chunk_shape: &S,
    sampling_rate: usize,
    noise_buffer: &mut [f64],
) {
    let chunk_size_i = chunk_shape.as_array()[0] as i32;
    for z in 1..(chunk_size_i - 1) {
        for y in 1..(chunk_size_i - 1) {
            for x in 1..(chunk_size_i - 1) {
                let idx = chunk_shape.linearize([x, y, z]) as usize;
                if !noise_buffer[idx].is_nan() {
                    continue;
                }

                noise_buffer[idx] = tri_lerp(idx, chunk_shape, sampling_rate, noise_buffer)
            }
        }
    }
}

fn tri_lerp<S: Shape<3, Coord = i32>>(
    idx: usize,
    chunk_shape: &S,
    sampling_rate: usize,
    noise_buffer: &mut [f64],
) -> f64 {
    let sampling_rate = sampling_rate as i32;
    let [x, y, z] = chunk_shape.delinearize(idx as i32);
    let [x0, y0, z0] = [x, y, z].map(|s| ((s - 1) / sampling_rate) * sampling_rate + 1); // compensate for padding, 
    let [x1, y1, z1] = [x0, y0, z0].map(|s| s + sampling_rate);
    let [tx, ty, tz] =
        [[x, x0], [y, y0], [z, z0]].map(|[s, s0]| (s as f64 - s0 as f64) / sampling_rate as f64);

    let c000 = noise_buffer[chunk_shape.linearize([x0, y0, z0]) as usize];
    let c001 = noise_buffer[chunk_shape.linearize([x0, y0, z1]) as usize];
    let c010 = noise_buffer[chunk_shape.linearize([x0, y1, z0]) as usize];
    let c011 = noise_buffer[chunk_shape.linearize([x0, y1, z1]) as usize];
    let c100 = noise_buffer[chunk_shape.linearize([x1, y0, z0]) as usize];
    let c101 = noise_buffer[chunk_shape.linearize([x1, y0, z1]) as usize];
    let c110 = noise_buffer[chunk_shape.linearize([x1, y1, z0]) as usize];
    let c111 = noise_buffer[chunk_shape.linearize([x1, y1, z1]) as usize];

    let c00 = lerp(c000, c100, tx);
    let c01 = lerp(c001, c101, tx);
    let c10 = lerp(c010, c110, tx);
    let c11 = lerp(c011, c111, tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);

    lerp(c0, c1, tz)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

// Chunk generation occurs with positive strides. Given chunk_size = 32...
//  chunk_key at (0,0,0) => chunk_min_voxel (0,0,0) => (31,31,31)
//  chunk_key at (1,1,1) => chunk_min_voxel (32,32,32) => (63,63,63)
//  chunk_key at (-1,-1,-1) => chunk_min_voxel (-32,-32,-32) => (-1,-1,-1)
pub fn generate_chunk<S: Shape<3, Coord = i32>>(
    wv: &IVec3,
    seed: u32,
    chunk_shape: S,
    new_chunk: &mut [Block],
) {
    // let t1 = std::time::Instant::now();
    let mut noise = HybridMulti::<Perlin>::new(seed);
    noise.octaves = 5;
    noise.frequency = 1.1;
    noise.lacunarity = 2.8;
    noise.persistence = 0.4;

    let mut cache = HashMap::<(i32, i32), f64>::new();
    let chunk_size_i = chunk_shape.as_array()[0] as i32;

    for z in 0..chunk_size_i {
        for y in 0..chunk_size_i {
            for x in 0..chunk_size_i {
                let idx = (x + y * chunk_size_i + z * chunk_size_i * chunk_size_i) as usize;
                let pos = IVec3::new(x + wv.x, y + wv.y, z + wv.z);
                if pos.y < -20 {
                    new_chunk[idx] = Block::Dirt;
                    continue;
                }

                let x_z = (pos.x, pos.z);
                let pos_64 = Vector3 {
                    x: pos.x as f64,
                    y: pos.y as f64,
                    z: pos.z as f64,
                };

                // If y is less than the noise sample, we will set the voxel to solid
                let is_ground = pos_64.y
                    < match cache.get(&x_z) {
                        Some(sample) => *sample,
                        None => {
                            let sample = noise.get([pos_64.x / 1000.0, pos_64.z / 1000.0]) * 50.0;
                            cache.insert(x_z, sample);
                            sample
                        }
                    };

                if is_ground {
                    new_chunk[idx] = Block::Grass;
                } else {
                    new_chunk[idx] = Block::None;
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_mesh::ndshape::{ConstShape, ConstShape3i32};
    type ChunkShape4 = ConstShape3i32<4, 4, 4>;
    type ChunkShape10 = ConstShape3i32<10, 10, 10>;

    #[test]
    fn tri_lerp_origin_points() {
        let sampling_rate = 2;
        let chunk_shape = ChunkShape4 {};
        let mut noise_buffer = [f64::NAN; ChunkShape4::SIZE as usize];

        noise_buffer[chunk_shape.linearize([1, 1, 1]) as usize] = 0.0; //c000
        noise_buffer[chunk_shape.linearize([3, 1, 1]) as usize] = 1.0; //c100
        noise_buffer[chunk_shape.linearize([1, 3, 1]) as usize] = 2.0; //c010
        noise_buffer[chunk_shape.linearize([1, 1, 3]) as usize] = 4.0; //c001
        noise_buffer[chunk_shape.linearize([3, 3, 1]) as usize] = 3.0; //c110
        noise_buffer[chunk_shape.linearize([3, 1, 3]) as usize] = 5.0; //c101
        noise_buffer[chunk_shape.linearize([1, 3, 3]) as usize] = 6.0; //c011
        noise_buffer[chunk_shape.linearize([3, 3, 3]) as usize] = 7.0; //c111

        let x_axis = tri_lerp(
            chunk_shape.linearize([2, 1, 1]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(x_axis, 0.5);

        let y_axis = tri_lerp(
            chunk_shape.linearize([1, 2, 1]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(y_axis, 1.0);

        let z_axis = tri_lerp(
            chunk_shape.linearize([1, 1, 2]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(z_axis, 2.0);

        let diag = tri_lerp(
            chunk_shape.linearize([2, 2, 2]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(diag, 3.5);
    }

    #[test]
    fn tri_lerp_offset_points() {
        let sampling_rate = 4;
        let chunk_shape = ChunkShape10 {};
        let mut noise_buffer = [f64::NAN; ChunkShape10::SIZE as usize];

        noise_buffer[chunk_shape.linearize([5, 5, 5]) as usize] = 0.0; //c000
        noise_buffer[chunk_shape.linearize([9, 5, 5]) as usize] = 4.0; //c100
        noise_buffer[chunk_shape.linearize([5, 9, 5]) as usize] = 8.0; //c010
        noise_buffer[chunk_shape.linearize([5, 5, 9]) as usize] = 16.0; //c001
        noise_buffer[chunk_shape.linearize([9, 9, 5]) as usize] = 12.0; //c110
        noise_buffer[chunk_shape.linearize([9, 5, 9]) as usize] = 20.0; //c101
        noise_buffer[chunk_shape.linearize([5, 9, 9]) as usize] = 24.0; //c011
        noise_buffer[chunk_shape.linearize([9, 9, 9]) as usize] = 28.0; //c111

        let x_axis = tri_lerp(
            chunk_shape.linearize([6, 5, 5]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(x_axis, 1.0);

        let y_axis = tri_lerp(
            chunk_shape.linearize([5, 6, 5]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(y_axis, 2.0);

        let z_axis = tri_lerp(
            chunk_shape.linearize([5, 5, 6]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(z_axis, 4.0);

        let diag_1 = tri_lerp(
            chunk_shape.linearize([6, 6, 6]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(diag_1, 7.0);

        let diag_2 = tri_lerp(
            chunk_shape.linearize([7, 7, 7]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(diag_2, 14.0);

        let diag_3 = tri_lerp(
            chunk_shape.linearize([8, 8, 8]) as usize,
            &chunk_shape,
            sampling_rate,
            &mut noise_buffer,
        );
        assert_eq!(diag_3, 21.0);
    }

    #[test]
    fn generate_chunk_noise() {
        let chunk_wv = IVec3::new(0, 0, 0);
        let chunk_shape = ChunkShape4 {};
        let sampling_rate = 2;
        let noise = HybridMulti::<Perlin>::new(1234);
        let mut noise_buffer = [f64::NAN; ChunkShape4::SIZE as usize];

        super::generate_chunk_noise(
            &chunk_wv,
            &chunk_shape,
            sampling_rate,
            1.0,
            &noise,
            &mut noise_buffer,
        );

        println!("{:?}", noise_buffer);

        let [x, y, z] = [1, 1, 1];
        let c000 = noise_buffer[chunk_shape.linearize([x, y, z]) as usize];
        assert_eq!(c000, noise.get([(x - 1) as f64, (z - 1) as f64]));

        let real_value_points = [
            [1, 1, 1],
            [1, 1, 3],
            [1, 3, 1],
            [1, 3, 3],
            [3, 1, 1],
            [3, 1, 3],
            [3, 3, 1],
            [3, 3, 3],
        ];

        for [x, y, z] in real_value_points {
            let expected = noise.get([(x - 1) as f64, (z - 1) as f64]);
            let actual = noise_buffer[chunk_shape.linearize([x, y, z]) as usize];
            assert_eq!(expected, actual);
        }

        let interpolated_value_points = [
            // [1, 1, 1], origin => noise value
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 1],
            [2, 2, 2],
        ];

        for [x, y, z] in interpolated_value_points {
            let idx = chunk_shape.linearize([x, y, z]) as usize;
            let expected = tri_lerp(idx, &chunk_shape, sampling_rate, &mut noise_buffer);
            let actual = noise_buffer[idx];
            assert_eq!(expected, actual);
        }
    }
}
