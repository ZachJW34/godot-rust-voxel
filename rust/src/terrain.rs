use std::collections::HashMap;

use crate::voxel::Block;
use block_mesh::ndshape::Shape;
use glam::IVec3;
use noise::{HybridMulti, NoiseFn, Perlin, Vector3};

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
                if pos.y < 1 {
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
