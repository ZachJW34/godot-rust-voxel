use block_mesh::{GreedyQuadsBuffer, MergeVoxel, Voxel, greedy_quads, ndshape::ConstShape3u32};
use cgmath::{Point3, Vector3};
use godot::{classes::StandardMaterial3D, global::godot_print, obj::Gd};

pub const CHUNK_SIZE: usize = 4;
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
type ChunkShape = ConstShape3u32<16, 16, 16>;

#[derive(Clone, PartialEq, Eq, Copy)]
pub enum Block {
    Grass,
    None,
    Stone,
}

impl Voxel for Block {
    fn get_visibility(&self) -> block_mesh::VoxelVisibility {
        block_mesh::VoxelVisibility::Opaque
    }
}

impl MergeVoxel for Block {
    type MergeValue = Self;

    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

impl Block {
    pub fn atlas_uv(&self) -> (u8, u8) {
        match self {
            Block::Grass => (0, 0),
            Block::Stone => (0, 1),
            Block::None => panic!("Can't get uv for None"),
        }
    }
}

pub trait ToMaterial {
    fn to_material(&self) -> Gd<StandardMaterial3D>;
}

pub struct VoxelWorld {
    pub x_chunks: usize,
    pub y_chunks: usize,
    pub z_chunks: usize,
    pub grid: Vec<Block>,
    total_chunks: usize,
    max_x: usize,
    max_y: usize,
    max_z: usize,
}

impl VoxelWorld {
    pub fn init(x_chunks: usize, y_chunks: usize, z_chunks: usize) -> Self {
        let total_blocks = x_chunks * y_chunks * z_chunks * CHUNK_VOLUME;
        godot_print!("[VoxelWorld] (init) - total_blocks: {total_blocks}");
        Self {
            x_chunks,
            y_chunks,
            z_chunks,
            grid: vec![Block::None; total_blocks],
            total_chunks: x_chunks * y_chunks * z_chunks,
            max_x: x_chunks * CHUNK_SIZE,
            max_y: y_chunks * CHUNK_SIZE,
            max_z: z_chunks * CHUNK_SIZE,
        }
    }

    pub fn point_to_idx(&self, point: &Point3<usize>) -> Option<usize> {
        if (point.x >= self.max_x) || (point.y >= self.max_y) || (point.z >= self.max_z) {
            return None;
        }

        let idx = point.x + point.y * self.max_x + point.z * self.max_x * self.max_y;

        if idx >= self.grid.len() {
            None
        } else {
            Some(idx)
        }
    }

    pub fn idx_to_point(&self, idx: usize) -> Option<Point3<usize>> {
        if idx >= self.grid.len() {
            return None;
        }

        let world_width = self.x_chunks * CHUNK_SIZE;
        let world_height = self.y_chunks * CHUNK_SIZE;

        let x = idx % world_width;
        let y = (idx / world_width) % world_height;
        let z = idx / (world_width * world_height);

        Some(Point3 { x, y, z })
    }

    pub fn add_block(&mut self, point: &Point3<usize>, new_block: Block) -> Block {
        let idx = self.point_to_idx(point).unwrap();
        std::mem::replace(&mut self.grid[idx], new_block)
    }

    pub fn remove_block(&mut self, point: &Point3<usize>) -> Block {
        let idx = self.point_to_idx(&point).unwrap();
        std::mem::replace(&mut self.grid[idx], Block::None)
    }

    pub fn mesh_chunks(&mut self) -> Vec<(Vec<ModelVertex>, Vec<i32>)> {
        let mut chunks: Vec<(Vec<ModelVertex>, Vec<i32>)> = vec![];

        for chunk in 0..self.total_chunks {
            let idx = chunk * CHUNK_VOLUME;
            if let Some(chunk) = self.mesh_chunk(idx) {
                chunks.push(chunk)
            }
        }

        chunks
    }

    pub fn mesh_chunk(&self, start_idx: usize) -> Option<(Vec<ModelVertex>, Vec<i32>)> {
        let mut vertices: Vec<ModelVertex> = vec![];
        let mut indices: Vec<i32> = vec![];

        for idx in start_idx..(start_idx + CHUNK_VOLUME) {
            let block = &self.grid[idx];
            if *block == Block::None {
                continue;
            }

            let cur_point = self.idx_to_point(idx).unwrap();

            for face in ALL_FACES {
                let adj_block = offset_point_safe(&cur_point, &face.normal.into())
                    .and_then(|p| self.point_to_idx(&p))
                    .and_then(|idx| Some(&self.grid[idx]))
                    .or(Some(&Block::None))
                    .unwrap();

                if *adj_block == Block::None {
                    generate_face(
                        &face,
                        &mut vertices,
                        &mut indices,
                        cur_point,
                        block.atlas_uv(),
                    );
                }
            }
        }

        if vertices.len() == 0 {
            None
        } else {
            Some((vertices, indices))
        }
    }

    pub fn mesh_chunks_greedy(&self) {
        let mut buffer = GreedyQuadsBuffer::new(self.grid.len());
        // greedy_quads(&self.grid, &ChunkShape , min, max, faces, output);
    }
}

fn offset_point_safe(p: &Point3<usize>, v: &Vector3<f32>) -> Option<Point3<usize>> {
    let p = Vector3::new(p.x as f32, p.y as f32, p.z as f32);
    let Vector3 { x, y, z } = p + v;

    if x < 0. || y < 0. || z < 0. {
        return None;
    }

    Some(Point3 {
        x: x as usize,
        y: y as usize,
        z: z as usize,
    })
}

#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

// 16 x 16 texture atlas
const ATLAS_STEP: f32 = 0.5;
// const ATLAS_ADJ: f32 = ATLAS_STEP * ATLAS_STEP;
const DEFAULT_TEX_COORDS: [[f32; 2]; 4] = [
    [0.0, 0.0],
    [1.0 / 16.0, 0.0],
    [1.0 / 16.0, 1.0 / 16.0],
    [0.0, 1.0 / 16.0],
];

// impl Block {
//     pub fn get_atlas_uv(&self, face: &Face, above: Option<Block>) -> (u8, u8) {
//         match self {
//             Block::Grass => match face {
//                 Face::Top => (0, 0),
//                 Face::Bottom => (2, 0),
//                 _ => {
//                     if let Some(Block::GRASS) = above {
//                         (2, 0)
//                     } else {
//                         (3, 0)
//                     }
//                 }
//             },
//             Block::STONE => (0, 1),
//             _ => panic!("No texture available for {self}"),
//         }
//     }
// }

pub enum Face {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

pub struct BlockFace {
    pub face: Face,
    pub vertices: [[f32; 3]; 4],
    pub normal: [f32; 3],
    pub tex_coords: [[f32; 2]; 4],
}

pub const FRONT_FACE: BlockFace = BlockFace {
    face: Face::Front,
    vertices: [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    normal: [0.0, 0.0, 1.0],
    tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
};

pub const BACK_FACE: BlockFace = BlockFace {
    face: Face::Back,
    vertices: [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ],
    normal: [0.0, 0.0, -1.0],
    tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
};

pub const LEFT_FACE: BlockFace = BlockFace {
    face: Face::Left,
    vertices: [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    normal: [-1.0, 0.0, 0.0],
    tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
};

pub const RIGHT_FACE: BlockFace = BlockFace {
    face: Face::Right,
    vertices: [
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    normal: [1.0, 0.0, 0.0],
    tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
};

pub const TOP_FACE: BlockFace = BlockFace {
    face: Face::Top,
    vertices: [
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    normal: [0.0, 1.0, 0.0],
    tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
};

pub const BOTTOM_FACE: BlockFace = BlockFace {
    face: Face::Bottom,
    vertices: [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    normal: [0.0, -1.0, 0.0],
    tex_coords: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
};

pub const ALL_FACES: [BlockFace; 6] = [
    FRONT_FACE,
    BACK_FACE,
    LEFT_FACE,
    RIGHT_FACE,
    TOP_FACE,
    BOTTOM_FACE,
];

pub fn generate_face(
    face: &BlockFace,
    vertices: &mut Vec<ModelVertex>,
    indices: &mut Vec<i32>,
    offset: Point3<usize>,
    texture: (u8, u8),
) {
    let start_index = vertices.len() as i32;
    let Point3 { x, y, z } = offset;
    let new_indices = [
        start_index,
        start_index + 1,
        start_index + 2,
        start_index,
        start_index + 2,
        start_index + 3,
    ];
    let mut new_vertices = vec![];
    let tex_coords = get_tex_coords_from_atlas(texture.0, texture.1);

    for (i, &position) in face.vertices.iter().enumerate() {
        new_vertices.push(ModelVertex {
            position: [
                position[0] + x as f32,
                position[1] + y as f32,
                position[2] + z as f32,
            ],
            normal: face.normal,
            tex_coords: tex_coords[i],
        });
    }
    // println!("New vertices: {new_vertices:?}");
    // println!("indices: {indices:?}");
    vertices.extend_from_slice(&new_vertices);
    indices.extend_from_slice(&new_indices);
}

fn get_tex_coords_from_atlas(x: u8, y: u8) -> [[f32; 2]; 4] {
    let (xmin, ymin) = (ATLAS_STEP * x as f32, ATLAS_STEP * y as f32);
    let (xmax, ymax) = (xmin + ATLAS_STEP, ymin + ATLAS_STEP);
    let (xmin, ymin) = (xmin, ymin);
    let (xmax, ymax) = (xmax, ymax);

    return [[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]];
}
