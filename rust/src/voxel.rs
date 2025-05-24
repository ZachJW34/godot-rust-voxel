use cgmath::Point3;
use godot::{classes::StandardMaterial3D, obj::Gd};

pub const CHUNK_SIZE: usize = 32;
pub const PLANE_SIZE: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const GRID_SIZE: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone, PartialEq)]
pub enum Block {
    Grass,
    None,
    Stone,
}

pub trait ToMaterial {
    fn to_material(&self) -> Gd<StandardMaterial3D>;
}

pub struct VoxelWorld {
    // num_chunks_x: usize,
    // num_chunks_y: usize,
    // num_chunks_z: usize,
    pub grid: Vec<Block>,
}

impl VoxelWorld {
    pub fn init() -> Self {
        Self {
            // num_chunks_x,
            // num_chunks_y,
            // num_chunks_z,
            grid: vec![Block::None; GRID_SIZE],
        }
    }

    pub fn point_to_idx(&self, Point3 { x, y, z }: &Point3<usize>) -> usize {
        return x + y * CHUNK_SIZE + z * PLANE_SIZE;
    }

    pub fn idx_to_point(&self, idx: usize) -> Point3<usize> {
        return Point3 {
            x: idx % CHUNK_SIZE,
            y: (idx / CHUNK_SIZE) % CHUNK_SIZE,
            z: idx / (PLANE_SIZE),
        };
    }

    pub fn add_block(&mut self, point: &Point3<usize>, new_block: Block) -> Block {
        let idx = self.point_to_idx(point);
        std::mem::replace(&mut self.grid[idx], new_block)
    }

    pub fn remove_block(&mut self, point: &Point3<usize>) -> Block {
        let idx = self.point_to_idx(&point);
        std::mem::replace(&mut self.grid[idx], Block::None)
    }

    pub fn mesh_chunk(&self, start_idx: usize) {
        for idx in start_idx..(start_idx + GRID_SIZE) {
            let block = &self.grid[idx];
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
    pub position: [f32; 3],
    // pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

// 16 x 16 texture atlas
const ATLAS_STEP: f32 = 1.0 / 16.0;
const ATLAS_ADJ: f32 = ATLAS_STEP * ATLAS_STEP;
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
    indices: &mut Vec<u32>,
    offset: Option<Point3<usize>>,
    // texture: Option<(u8, u8)>,
) {
    let start_index = vertices.len() as u32;
    let Point3 { x, y, z } = offset.unwrap_or(Point3::new(0, 0, 0));
    let new_indices = [
        start_index,
        start_index + 1,
        start_index + 2,
        start_index,
        start_index + 2,
        start_index + 3,
    ];
    let mut new_vertices = vec![];
    // let tex_coords = if let Some((x, y)) = texture {
    //     get_tex_coords_from_atlas(x, y)
    // } else {
    //     DEFAULT_TEX_COORDS
    // };

    for (i, &position) in face.vertices.iter().enumerate() {
        new_vertices.push(ModelVertex {
            position: [
                position[0] + x as f32,
                position[1] + y as f32,
                position[2] + z as f32,
            ],
            normal: face.normal,
            // tex_coords: tex_coords[i],
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
    let (xmin, ymin) = (xmin + ATLAS_ADJ, ymin + ATLAS_ADJ);
    let (xmax, ymax) = (xmax - ATLAS_ADJ, ymax - ATLAS_ADJ);

    return [[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]];
}
