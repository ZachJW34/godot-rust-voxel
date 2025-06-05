use std::fs::OpenOptions;
use std::io::Write;
use std::{collections::HashMap, sync::Arc};

use block_mesh::{
    GreedyQuadsBuffer, MergeVoxel, QuadBuffer, QuadCoordinateConfig, RIGHT_HANDED_Y_UP_CONFIG,
    UnitQuadBuffer, Voxel, greedy_quads,
    ndshape::{ConstShape, ConstShape3i32, ConstShape3u32},
    visible_block_faces,
};
use crossbeam::queue::{ArrayQueue, SegQueue};
use dashmap::DashMap;
use glam::IVec3;
use godot::global::godot_print;
use rayon::{
    ThreadPool, ThreadPoolBuilder,
    iter::{IntoParallelRefIterator, ParallelIterator},
};

use crate::terrain::generate_chunk;

pub const CHUNK_SIZE_I: i32 = 32;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE_I * CHUNK_SIZE_I * CHUNK_SIZE_I) as usize;
pub const CHUNK_SIZE_I_PADDED: i32 = 32 + 2;
// pub const CHUNK_SIZE_U: u32 = CHUNK_SIZE_I as u32;
pub const CHUNK_SIZE_U_PADDED: u32 = CHUNK_SIZE_I_PADDED as u32;
// pub type ChunkShapeU =
//     ConstShape3u32<CHUNK_SIZE_U_PADDED, CHUNK_SIZE_U_PADDED, CHUNK_SIZE_U_PADDED>;

pub struct VoxelWorld {
    chunks: Arc<DashMap<IVec3, Chunk32>>,
    pub render_queue: Arc<DashMap<IVec3, ChunkMesh>>,
    mesh_thread_pool: Arc<ThreadPool>,
    radius_h: usize,
    radius_v: usize,
    mesh_queue: ArrayQueue<IVec3>,
    meshing_queue: Arc<SegQueue<IVec3>>,
}

impl VoxelWorld {
    pub fn new(radius_h: usize, radius_v: usize) -> Self {
        Self {
            chunks: Arc::new(DashMap::new()),
            render_queue: Arc::new(DashMap::new()),
            mesh_thread_pool: Arc::new(ThreadPoolBuilder::new().num_threads(8).build().unwrap()),
            radius_h,
            radius_v,
            mesh_queue: ArrayQueue::new(2 * radius_h * 2 * radius_h * 2 * radius_v),
            meshing_queue: Arc::new(SegQueue::new()),
        }
    }

    pub fn add_change(&mut self, wv: &IVec3, new_block: Block) -> Block {
        let chunk_v = wv_to_chunk_v(wv);
        if !self.chunks.contains_key(&chunk_v) {
            self.chunks.insert(chunk_v, Chunk32::empty());
        }

        let mut chunk = self.chunks.get_mut(&chunk_v).unwrap();
        let old_block = chunk.add_change(wv, new_block);
        self.mesh_queue.force_push(chunk_v);

        old_block
    }

    pub fn reset_mesh(&mut self) {
        self.render_queue.clear();
        self.chunks.alter_all(|_, mut chunk| {
            chunk.pipeline = Pipeline::NeedsFastMesh;
            chunk
        });
    }

    pub fn remove_generated(&mut self) {
        self.render_queue.clear();
        self.chunks.alter_all(|_, mut chunk| {
            chunk.voxels = Chunk32::empty().voxels;
            chunk.generated = false;
            if chunk.changes.len() > 0 {
                chunk.pipeline = Pipeline::NeedsFastMesh;
            }

            chunk
        });
    }

    pub fn queue_chunks_to_mesh(&mut self, camera_pos: &IVec3) {
        let camera_chunk_v = wv_to_chunk_v(camera_pos);

        let radius_h = self.radius_h as i32;
        let radius_v = self.radius_v as i32;

        let mut chunk_vs = Vec::with_capacity(self.radius_h * self.radius_h * self.radius_v);

        for x in -radius_h..radius_h {
            for y in -radius_v..radius_v {
                for z in -radius_h..radius_h {
                    let wv = IVec3 {
                        x: camera_pos.x + (x * CHUNK_SIZE_I),
                        y: camera_pos.y + (y * CHUNK_SIZE_I),
                        z: camera_pos.z + (z * CHUNK_SIZE_I),
                    };

                    let chunk_v = wv_to_chunk_v(&wv);
                    chunk_vs.push(chunk_v);
                    // let contains_key = self.chunks.contains_key(&chunk_v);
                    // if !contains_key {
                    //     let mut new_chunk = Chunk32::empty();
                    //     new_chunk.pipeline = Pipeline::NeedsFastMesh;

                    //     self.chunks.insert(chunk_v, new_chunk);
                    // }

                    // if contains_key && self.chunks.get(&chunk_v).unwrap().pipeline == Pipeline::None
                    // {
                    //     let mut chunk = self.chunks.get_mut(&chunk_v).unwrap();
                    //     chunk.pipeline = Pipeline::NeedsFastMesh
                    // }
                }
            }
        }

        chunk_vs.sort_by_key(|chunk_v| (camera_chunk_v - chunk_v).length_squared());
        chunk_vs.into_iter().for_each(|chunk| {
            self.mesh_queue.force_push(chunk);
        });
    }

    pub fn mesh_queued_chunks(&mut self, camera_pos: &IVec3, procedural: bool) {
        let t1 = std::time::Instant::now();

        // let mut chunks_to_mesh: Vec<_> = self
        //     .chunks
        //     .iter()
        //     .filter_map(|entry| {
        //         matches!(entry.pipeline, Pipeline::NeedsFastMesh).then_some(entry.key().clone())
        //     })
        //     .collect();

        // chunks_to_mesh.sort_by_key(|cp| (cp * CHUNK_SIZE_I - camera_pos).length_squared());

        // if chunks_to_mesh.len() > 0 {
        //     godot_print!(
        //         "[VoxelWorld] (mesh_queued_chunks) - t={:?} len={}",
        //         t1.elapsed(),
        //         chunks_to_mesh.len()
        //     );
        // }

        // let chunk_to_mesh = chunks_to_mesh.get(0);
        // if chunk_to_mesh.is_none() {
        //     return;
        // }
        // let chunk_to_mesh = chunk_to_mesh.unwrap().clone();
        // let mut chunk = self.chunks.get_mut(&chunk_to_mesh).unwrap();
        // chunk.pipeline = Pipeline::Meshing;

        // drop(chunk);

        // let meshing_count = self
        //     .chunks
        //     .iter()
        //     .filter(|entry| matches!(entry.pipeline, Pipeline::Meshing))
        //     .count();
        let meshing_count = self.meshing_queue.len();

        let available_tasks = self
            .mesh_thread_pool
            .current_num_threads()
            .saturating_sub(meshing_count);

        if available_tasks == 0 {
            return;
        }

        let mut chunks_to_mesh = vec![];
        for _ in 0..self.mesh_queue.len() {
            if chunks_to_mesh.len() >= available_tasks * 2 {
                break;
            }
            if let Some(chunk_v) = self.mesh_queue.pop() {
                if !self.chunks.contains_key(&chunk_v) {
                    let mut new_chunk = Chunk32::empty();
                    new_chunk.pipeline = Pipeline::NeedsFastMesh;
                    self.chunks.insert(chunk_v, new_chunk);
                }

                if self.chunks.get(&chunk_v).unwrap().pipeline == Pipeline::NeedsFastMesh {
                    chunks_to_mesh.push(chunk_v);
                }
            }
        }

        if chunks_to_mesh.len() > 0 {
            // godot_print!(
            //     "Available tasks: {available_tasks} meshing_count: {meshing_count} threads: {} meshing_queue.len: {} chunks_to_mesh: {:?}",
            //     self.mesh_thread_pool.current_num_threads(),
            //     self.mesh_queue.len(),
            //     chunks_to_mesh
            // );
        }

        let camera_chunk_v = wv_to_chunk_v(camera_pos);
        for chunk_v in &chunks_to_mesh {
            let mut chunk = self.chunks.get_mut(chunk_v).unwrap();
            chunk.pipeline = Pipeline::Meshing;
            self.meshing_queue.push(*chunk_v);
        }

        let render_queue = self.render_queue.clone();
        let chunks = self.chunks.clone();
        let pool = self.mesh_thread_pool.clone();
        let meshing_queue = self.meshing_queue.clone();

        godot::task::spawn(async move {
            for chunk_v in chunks_to_mesh {
                let chunks = chunks.clone();
                let render_queue = render_queue.clone();
                let meshing_queue = meshing_queue.clone();
                pool.spawn(move || {
                    let chunk = chunks.get(&chunk_v).unwrap();
                    let mut working_chunk = chunk.clone();
                    drop(chunk);

                    if procedural && !working_chunk.generated {
                        let chunk_wv = chunk_v_to_wv(&chunk_v);
                        generate_chunk(
                            &chunk_wv,
                            1234,
                            working_chunk.shape(),
                            &mut working_chunk.voxels,
                        );
                        working_chunk.generated = true;
                    }

                    let chunk = working_chunk.clone();
                    chunks.insert(chunk_v, chunk);

                    let merged_voxels = working_chunk.merge_changes();
                    // let mut buffer = UnitQuadBuffer::new();
                    // visible_block_faces(
                    //     &merged_voxels.voxels,
                    //     &ChunkShapeU {},
                    //     [0; 3],
                    //     [33; 3],
                    //     &RIGHT_HANDED_Y_UP_CONFIG.faces,
                    //     &mut buffer,
                    // );

                    let mut buffer = GreedyQuadsBuffer::new(CHUNK_VOLUME * 6);
                    greedy_quads(
                        &merged_voxels.voxels,
                        &merged_voxels.shape(),
                        [0; 3],
                        [33; 3],
                        &RIGHT_HANDED_Y_UP_CONFIG.faces,
                        &mut buffer,
                    );

                    let mesh = if buffer.quads.num_quads() == 0 {
                        ChunkMesh {
                            indices: vec![],
                            positions: vec![],
                            normals: vec![],
                            uvs: vec![],
                            layers: vec![],
                            tangents: vec![],
                        }
                    } else {
                        let chunk_offset = [
                            (chunk_v.x * CHUNK_SIZE_I) as f32,
                            (chunk_v.y * CHUNK_SIZE_I) as f32,
                            (chunk_v.z * CHUNK_SIZE_I) as f32,
                        ];

                        build_mesh_from_quads(
                            buffer.quads,
                            &merged_voxels,
                            &RIGHT_HANDED_Y_UP_CONFIG,
                            &chunk_offset,
                        )
                    };

                    render_queue.insert(chunk_v, mesh);
                    meshing_queue.pop();

                    // let mut file = OpenOptions::new()
                    //     .create(true)
                    //     .append(true)
                    //     .open("log.txt")
                    //     .unwrap();

                    // writeln!(file, "thread t={:?}", t1.elapsed()).unwrap();
                })
            }
        });
    }
}

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
    pub fn layer_idx(&self) -> f32 {
        match self {
            Block::Grass => 0.0,
            Block::Dirt => 1.0,
            Block::Stone => 2.0,
            Block::Snow => 3.0,
            Block::None => panic!("Can't get layer_idx for None"),
        }
    }
}

impl From<i32> for Block {
    fn from(value: i32) -> Self {
        match value {
            0 => Block::None,
            1 => Block::Dirt,
            2 => Block::Grass,
            3 => Block::Stone,
            4 => Block::Snow,
            _ => panic!("Don't have a Block::from({value}) implemented"),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Pipeline {
    None,
    NeedsFastMesh,
    Meshing,
    Meshed,
    Rendered,
}

pub type Chunk32Shape = ConstShape3i32<32, 32, 32>;

#[derive(Clone)]
pub struct Chunk32 {
    voxels: [Block; 32 * 32 * 32],
    changes: HashMap<IVec3, Block>,
    generated: bool,
    pipeline: Pipeline,
}

impl Chunk32 {
    fn empty() -> Self {
        Self {
            voxels: [Block::None; 32 * 32 * 32],
            changes: HashMap::new(),
            generated: false,
            pipeline: Pipeline::None,
        }
    }

    pub fn add_change(&mut self, wv: &IVec3, new_block: Block) -> Block {
        let voxel_v = wv_to_voxel_v(wv);
        let old_block = self.changes.insert(voxel_v, new_block).unwrap_or_else(|| {
            let idx = self.linearize(&voxel_v);
            godot_print!(
                "[VoxelWorld2] (add_change) - wv: {wv} => voxel_v: {voxel_v} => idx: {idx}"
            );
            self.voxels[idx]
        });

        self.pipeline = Pipeline::NeedsFastMesh;

        old_block
    }

    fn linearize(&self, voxel_v: &IVec3) -> usize {
        Chunk32Shape::linearize(voxel_v.to_array()) as usize
    }

    pub fn merge_changes(self) -> Chunk34 {
        let mut voxels = [Block::None; 34 * 34 * 34];

        let size = 32;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let voxel_v = IVec3::new(x, y, z);
                    let idx = self.linearize(&voxel_v);
                    let block = self
                        .changes
                        .get(&voxel_v)
                        .unwrap_or_else(|| &self.voxels[idx]);

                    // Offset by 1 to account for padding
                    let dst_x = (x + 1) as u32;
                    let dst_y = (y + 1) as u32;
                    let dst_z = (z + 1) as u32;
                    let dst_idx = Chunk34Shape::linearize([dst_x, dst_y, dst_z]) as usize;

                    voxels[dst_idx] = *block;
                }
            }
        }

        Chunk34 { voxels }
    }

    pub fn shape(&self) -> Chunk32Shape {
        Chunk32Shape {}
    }
}

pub type Chunk34Shape = ConstShape3u32<34, 34, 34>;
pub struct Chunk34 {
    voxels: [Block; 34 * 34 * 34],
}

impl Chunk34 {
    pub fn linearize(&self, p: [u32; 3]) -> usize {
        Chunk34Shape::linearize(p) as usize
    }

    pub fn shape(&self) -> Chunk34Shape {
        Chunk34Shape {}
    }
}

pub fn wv_to_chunk_v(wv: &IVec3) -> IVec3 {
    wv.div_euclid(IVec3::splat(CHUNK_SIZE_I))
}

fn chunk_v_to_wv(chunk_v: &IVec3) -> IVec3 {
    chunk_v * CHUNK_SIZE_I
}

fn wv_to_voxel_v(wv: &IVec3) -> IVec3 {
    wv.rem_euclid(IVec3::splat(CHUNK_SIZE_I))
}

#[derive(Debug)]
pub struct ChunkMesh {
    pub indices: Vec<u32>,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub layers: Vec<f32>,
    pub tangents: Vec<[f32; 4]>,
}

impl ChunkMesh {
    fn vert_to_idx(&self, vert: usize) -> usize {
        self.indices[3 * 3 + vert] as usize
    }
}

impl mikktspace::Geometry for ChunkMesh {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, _face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.vert_to_idx(vert)]
    }

    fn normal(&self, _face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.vert_to_idx(vert)]
    }

    fn tex_coord(&self, _face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.vert_to_idx(vert)]
    }

    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        _bi_tangent: [f32; 3],
        _f_mag_s: f32,
        _f_mag_t: f32,
        bi_tangent_preserves_orientation: bool,
        _face: usize,
        vert: usize,
    ) {
        let idx = self.vert_to_idx(vert);
        let w = if bi_tangent_preserves_orientation {
            1.0
        } else {
            -1.0
        };
        self.tangents[idx] = [tangent[0], tangent[1], tangent[2], w];
    }
}

fn build_mesh_from_quads(
    buffer: QuadBuffer,
    chunk: &Chunk34,
    config: &QuadCoordinateConfig,
    chunk_offset: &[f32; 3],
) -> ChunkMesh {
    let num_indices = buffer.num_quads() * 6;
    let num_vertices = buffer.num_quads() * 4;

    let mut indices = Vec::with_capacity(num_indices);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut uvs = Vec::with_capacity(num_vertices);
    let mut layers = Vec::with_capacity(num_vertices);
    let tangents = vec![[0.0; 4]; num_vertices];

    for (group, face) in buffer.groups.into_iter().zip(config.faces.into_iter()) {
        for quad in group.into_iter() {
            let mut new_indices = face.quad_mesh_indices(positions.len() as u32);
            let mut new_positions = face.quad_mesh_positions(&quad.into(), 1.0);
            for v in &mut new_positions {
                for i in 0..3 {
                    v[i] = v[i] - 1.0 + chunk_offset[i];
                }
            }
            let new_normals = face.quad_mesh_normals();
            let new_uvs = face.tex_coords(config.u_flip_face, true, &quad.into());

            let voxel_index = chunk.linearize(quad.minimum) as usize;
            let voxel_type = chunk.voxels[voxel_index];
            let layer_idx = voxel_type.layer_idx();
            let new_layers = &[layer_idx; 4];

            // TODO: This is fuarked, there is a winding problem
            new_indices.swap(1, 2);
            new_indices.swap(4, 5);

            indices.extend_from_slice(&new_indices);
            positions.extend_from_slice(&new_positions);
            normals.extend_from_slice(&new_normals);
            uvs.extend_from_slice(&new_uvs);
            layers.extend_from_slice(new_layers);
        }
    }

    ChunkMesh {
        indices,
        positions,
        normals,
        uvs,
        layers,
        tangents,
    }
}
