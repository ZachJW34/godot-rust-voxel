use block_mesh::{
    Axis, GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, RIGHT_HANDED_Y_UP_CONFIG,
    UnorientedQuad, Voxel, greedy_quads, ndshape::ConstShape3u32,
};
use cgmath::{Point3, Vector3};
use godot::{classes::StandardMaterial3D, global::godot_print, obj::Gd};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
pub const PADDING: usize = 1;
pub const CHUNK_SIZE_PADDED: usize = CHUNK_SIZE + PADDING + PADDING;
pub const CHUNK_VOLUME_PADDED: usize = CHUNK_SIZE_PADDED * CHUNK_SIZE_PADDED * CHUNK_SIZE_PADDED;
const PLANE_OFFSET: usize = PADDING * CHUNK_SIZE_PADDED * CHUNK_SIZE_PADDED;
const ROW_OFFSET: usize = PADDING * CHUNK_SIZE_PADDED;
const COLUMN_OFFSET: usize = PADDING;
const START_OFFSET: usize = COLUMN_OFFSET + ROW_OFFSET + PLANE_OFFSET;

#[derive(Clone, PartialEq, Eq, Copy, Debug)]
pub enum Block {
    None,
    Grass,
    Stone,
    Dirt,
    Snow,
}

impl Voxel for Block {
    fn get_visibility(&self) -> block_mesh::VoxelVisibility {
        match self {
            Block::None => block_mesh::VoxelVisibility::Empty,
            _ => block_mesh::VoxelVisibility::Opaque,
        }
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
            Block::Stone => (1, 0),
            Block::Dirt => (0, 1),
            Block::Snow => (1, 1),
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
        let total_blocks = x_chunks * y_chunks * z_chunks * CHUNK_VOLUME_PADDED;
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
        if point.x >= self.max_x || point.y >= self.max_y || point.z >= self.max_z {
            return None;
        }

        // Determine chunk
        let chunk_x = point.x / CHUNK_SIZE;
        let chunk_y = point.y / CHUNK_SIZE;
        let chunk_z = point.z / CHUNK_SIZE;
        let chunk_index =
            chunk_x + chunk_y * self.x_chunks + chunk_z * self.x_chunks * self.y_chunks;
        let chunk_offset = chunk_index * CHUNK_VOLUME_PADDED;

        let local_x: usize = point.x % CHUNK_SIZE;
        let local_y = point.y % CHUNK_SIZE;
        let local_z = point.z % CHUNK_SIZE;

        // Convert to padded chunk-local coordinates
        let local_index = (local_x + PADDING)
            + (local_y + PADDING) * CHUNK_SIZE_PADDED
            + (local_z + PADDING) * CHUNK_SIZE_PADDED * CHUNK_SIZE_PADDED;

        Some(chunk_offset + local_index)
    }

    pub fn idx_to_point(&self, idx: usize) -> Option<Point3<usize>> {
        if idx >= self.grid.len() {
            return None;
        }

        let chunk_index = idx / CHUNK_VOLUME_PADDED;
        let local_index = idx % CHUNK_VOLUME_PADDED;

        let chunk_x = chunk_index % self.x_chunks;
        let chunk_y = (chunk_index / self.x_chunks) % self.y_chunks;
        let chunk_z = chunk_index / (self.x_chunks * self.y_chunks);

        let lx = local_index % CHUNK_SIZE_PADDED;
        let ly = (local_index / CHUNK_SIZE_PADDED) % CHUNK_SIZE_PADDED;
        let lz = local_index / (CHUNK_SIZE_PADDED * CHUNK_SIZE_PADDED);

        // Remove padding offset
        if lx < PADDING || ly < PADDING || lz < PADDING {
            return None; // padding area, not a valid world voxel
        }

        Some(Point3 {
            x: chunk_x * CHUNK_SIZE + (lx - PADDING),
            y: chunk_y * CHUNK_SIZE + (ly - PADDING),
            z: chunk_z * CHUNK_SIZE + (lz - PADDING),
        })
    }

    pub fn add_block(&mut self, point: &Point3<usize>, new_block: Block) -> Block {
        let idx = self.point_to_idx(point).unwrap();
        // godot_print!("Putting {new_block:?} at {point:?} => {idx}");
        std::mem::replace(&mut self.grid[idx], new_block)
    }

    pub fn remove_block(&mut self, point: &Point3<usize>) -> Block {
        let idx = self.point_to_idx(&point).unwrap();
        std::mem::replace(&mut self.grid[idx], Block::None)
    }

    pub fn mesh_chunks(&mut self) -> Vec<(Vec<ModelVertex>, Vec<i32>)> {
        let t1 = std::time::Instant::now();
        let mut chunks = vec![];

        for chunk in 0..self.total_chunks {
            let idx: usize = chunk * CHUNK_VOLUME_PADDED;
            // godot_print!("Starting mesh of chunk={chunk} => idx={idx}");
            if let Some(chunk) = self.mesh_chunk(idx) {
                chunks.push(chunk)
            }
        }

        godot_print!("Total mesh chunk time: {:?}", t1.elapsed());

        chunks
    }

    // start_idx is the start of the padded chunk
    pub fn mesh_chunk(&self, start_idx: usize) -> Option<(Vec<ModelVertex>, Vec<i32>)> {
        // let t1 = std::time::Instant::now();

        let mut vertices: Vec<ModelVertex> = vec![];
        let mut indices: Vec<i32> = vec![];

        let start_idx = start_idx + START_OFFSET;
        let start_point = self.idx_to_point(start_idx).unwrap();
        // godot_print!("start_idx={start_idx} start_point={start_point:?}");

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let cur_point = start_point + Vector3::new(x, y, z);
                    let cur_idx = self.point_to_idx(&cur_point).unwrap();
                    let cur_voxel = &self.grid[cur_idx];

                    // godot_print!("{cur_point:?} ({cur_idx}) = {cur_voxel:?}");

                    if *cur_voxel == Block::None {
                        // godot_print!("  Skipping");
                        continue;
                    }

                    // godot_print!("  Meshing...");

                    for face in ALL_FACES {
                        let (adj_point, adj_point_raw) =
                            offset_point_safe(&cur_point, &face.normal.into());
                        let adj_voxel = adj_point
                            .and_then(|p| self.point_to_idx(&p))
                            .and_then(|idx| Some(&self.grid[idx]))
                            .or(Some(&Block::None))
                            .unwrap();
                        // let adj_voxel = offset_point_safe(&cur_point, &face.normal.into())
                        //     .and_then(|p| self.point_to_idx(&p))
                        //     .and_then(|idx| Some(&self.grid[idx]))
                        //     .or(Some(&Block::None))
                        //     .unwrap();

                        let msg = if *adj_voxel == Block::None {
                            format!("Meshing...")
                        } else {
                            format!("Skipping")
                        };

                        // godot_print!(
                        //     "   {:?} -> {:?} = {:?}: {}",
                        //     face.face,
                        //     adj_point_raw,
                        //     adj_voxel,
                        //     msg
                        // );

                        if *adj_voxel == Block::None {
                            // godot_print!(
                            //     "{:?} -> {:?} = {:?}: {:?} will be meshed",
                            //     cur_point,
                            //     adj_point,
                            //     adj_voxel,
                            //     face.face
                            // );
                            generate_face(
                                &face,
                                &mut vertices,
                                &mut indices,
                                cur_point,
                                cur_voxel.atlas_uv(),
                            );
                        }
                    }
                }
            }
        }

        // godot_print!("Mesh chunk time: {:?}", t1.elapsed());

        if vertices.len() == 0 {
            None
        } else {
            Some((vertices, indices))
        }
    }

    pub fn mesh_chunks_greedy(&self) {
        type ChunkShape = ConstShape3u32<34, 34, 34>;
        let mut buffer = GreedyQuadsBuffer::new(self.grid.len());
        let config = RIGHT_HANDED_Y_UP_CONFIG;
        greedy_quads(
            &self.grid,
            &ChunkShape {},
            [0; 3],
            [33; 3],
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
            &mut buffer,
        );

        let res = build_mesh_from_quads(&buffer, 1.0, config.u_flip_face, true);
    }
}

#[derive(Debug)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

fn build_mesh_from_quads(
    buffer: &GreedyQuadsBuffer,
    voxel_size: f32,
    u_flip_face: Axis,
    flip_v: bool,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut next_index = 0;

    for (group_index, group) in buffer.quads.groups.iter().enumerate() {
        let face = &RIGHT_HANDED_Y_UP_CONFIG.faces[group_index];

        for quad in group {
            // Get vertex positions in world space
            let positions = face.quad_mesh_positions(quad, voxel_size);
            let normals = face.quad_mesh_normals();
            let uvs = face.tex_coords(u_flip_face, flip_v, quad);

            // Build vertex data
            for i in 0..4 {
                vertices.push(Vertex {
                    position: positions[i],
                    normal: normals[i],
                    uv: uvs[i],
                });
            }

            // Add triangle indices
            let quad_indices = face.quad_mesh_indices(next_index);
            indices.extend_from_slice(&quad_indices);
            next_index += 4;
        }
    }

    (vertices, indices)
}

// struct Vertex {
//     position: [f32; 3],
//     normal: [f32; 3],
//     uv: [f32; 2],
// }

// fn build_mesh_from_quads(
//     buffer: &GreedyQuadsBuffer,
//     voxel_size: f32,
//     u_flip_face: Axis,
//     flip_v: bool,
//     atlas_layout: &TextureAtlasLayout, // Add atlas layout information
//     get_block_type: impl Fn(usize) -> u32, // Add block type lookup function
// ) -> (Vec<Vertex>, Vec<u32>) {
//     let mut vertices = Vec::new();
//     let mut indices = Vec::new();
//     let mut next_index = 0;

//     for (group_index, group) in buffer.quads.groups.iter().enumerate() {
//         let face = &RIGHT_HANDED_Y_UP_CONFIG.faces[group_index];

//         for (quad_index, quad) in group.iter().enumerate() {
//             // Get block type based on quad index
//             let block_type = get_block_type(quad_index);

//             // Calculate UV offset in the texture atlas based on block type
//             let atlas_index = block_type; // Or however you map block type to atlas index

//             let u_offset = (atlas_index % atlas_layout.columns) as f32 * atlas_layout.cell_width;
//             let v_offset = (atlas_index / atlas_layout.columns) as f32 * atlas_layout.cell_height;

//             // Get vertex positions in world space
//             let positions = face.quad_mesh_positions(quad, voxel_size);
//             let normals = face.quad_mesh_normals();
//             let base_uvs = face.tex_coords(u_flip_face, flip_v, quad);

//             // Build vertex data with atlas UVs
//             for i in 0..4 {
//                  let uv = [
//                     base_uvs[i][0] * atlas_layout.cell_width + u_offset,
//                     base_uvs[i][1] * atlas_layout.cell_height + v_offset,
//                     ];
//                  vertices.push(Vertex {
//                     position: positions[i],
//                     normal: normals[i],
//                     uv,
//                  });
//             }

//             // Add triangle indices
//             let quad_indices = face.quad_mesh_indices(next_index);
//             indices.extend_from_slice(&quad_indices);
//             next_index += 4;
//         }
//     }

//     (vertices, indices)
// }

// struct TextureAtlasLayout {
//     columns: u32,
//     rows: u32,
//     cell_width: f32,
//     cell_height: f32,
// }

fn offset_point_safe(p: &Point3<usize>, v: &Vector3<f32>) -> (Option<Point3<usize>>, Point3<f32>) {
    let p = Point3::new(p.x as f32, p.y as f32, p.z as f32);
    let t = p + v;

    if t.x < 0. || t.y < 0. || t.z < 0. {
        return (None, t);
    }

    (
        Some(Point3 {
            x: t.x as usize,
            y: t.y as usize,
            z: t.z as usize,
        }),
        t,
    )
}

#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub tangent: [f32; 4],
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

#[derive(Debug)]
pub enum Face {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug)]
pub struct BlockFace {
    pub face: Face,
    pub vertices: [[f32; 3]; 4],
    pub normal: [f32; 3],
    // pub tex_coords: [[f32; 2]; 4],
}

pub const FRONT_FACE: BlockFace = BlockFace {
    face: Face::Front,
    vertices: [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ],
    normal: [0.0, 0.0, 1.0],
    // tex_coords: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
};

pub const BACK_FACE: BlockFace = BlockFace {
    face: Face::Back,
    vertices: [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    normal: [0.0, 0.0, -1.0],
    // tex_coords: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
};

pub const LEFT_FACE: BlockFace = BlockFace {
    face: Face::Left,
    vertices: [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    normal: [-1.0, 0.0, 0.0],
    // tex_coords: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
};

pub const RIGHT_FACE: BlockFace = BlockFace {
    face: Face::Right,
    vertices: [
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    normal: [1.0, 0.0, 0.0],
    // tex_coords: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
};

pub const TOP_FACE: BlockFace = BlockFace {
    face: Face::Top,
    vertices: [
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    normal: [0.0, 1.0, 0.0],
    // tex_coords: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
};

pub const BOTTOM_FACE: BlockFace = BlockFace {
    face: Face::Bottom,
    vertices: [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    normal: [0.0, -1.0, 0.0],
    // tex_coords: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
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

    let tex_coords = get_tex_coords_from_atlas(texture.0, texture.1);
    let mut face_positions = [[0.0; 3]; 4];
    let mut face_uvs = [[0.0; 2]; 4];

    for i in 0..4 {
        face_positions[i] = [
            face.vertices[i][0] + x as f32,
            face.vertices[i][1] + y as f32,
            face.vertices[i][2] + z as f32,
        ];
        face_uvs[i] = tex_coords[i];
    }

    let tangent = calculate_tangent(&face_positions, &face_uvs, face.normal);

    let new_vertices: Vec<ModelVertex> = (0..4)
        .map(|i| ModelVertex {
            position: face_positions[i],
            normal: face.normal,
            tex_coords: face_uvs[i],
            tangent,
        })
        .collect();

    vertices.extend_from_slice(&new_vertices);
    indices.extend_from_slice(&new_indices);
}

fn calculate_tangent(pos: &[[f32; 3]; 4], uv: &[[f32; 2]; 4], normal: [f32; 3]) -> [f32; 4] {
    let (p0, p1, p2) = (pos[0], pos[1], pos[2]);
    let (uv0, uv1, uv2) = (uv[0], uv[1], uv[2]);

    let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

    let delta_uv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
    let delta_uv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

    let r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0]);
    let tangent = [
        r * (edge1[0] * delta_uv2[1] - edge2[0] * delta_uv1[1]),
        r * (edge1[1] * delta_uv2[1] - edge2[1] * delta_uv1[1]),
        r * (edge1[2] * delta_uv2[1] - edge2[2] * delta_uv1[1]),
    ];

    // Orthogonalize tangent against normal and normalize
    let dot = tangent[0] * normal[0] + tangent[1] * normal[1] + tangent[2] * normal[2];
    let tangent = [
        tangent[0] - normal[0] * dot,
        tangent[1] - normal[1] * dot,
        tangent[2] - normal[2] * dot,
    ];
    let len = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    let tangent = [tangent[0] / len, tangent[1] / len, tangent[2] / len];

    [tangent[0], tangent[1], tangent[2], 1.0] // Assume right-handed bitangent
}

fn get_tex_coords_from_atlas(x: u8, y: u8) -> [[f32; 2]; 4] {
    let (xmin, ymin) = (ATLAS_STEP * x as f32, ATLAS_STEP * y as f32);
    let (xmax, ymax) = (xmin + ATLAS_STEP, ymin + ATLAS_STEP);
    let (xmin, ymin) = (xmin, ymin);
    let (xmax, ymax) = (xmax, ymax);

    return [[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]];
}
