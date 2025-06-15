use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;
use std::{collections::HashMap, sync::Arc};

use block_mesh::{
    GreedyQuadsBuffer, MergeVoxel, QuadBuffer, QuadCoordinateConfig, RIGHT_HANDED_Y_UP_CONFIG,
    Voxel, greedy_quads,
    ndshape::{ConstShape, ConstShape3i32, ConstShape3u32},
};
use crossbeam::queue::{ArrayQueue, SegQueue};
use dashmap::DashMap;
use glam::{IVec3, Vec3};
use godot::global::godot_print;
use noise::{HybridMulti, Perlin};
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::terrain::generate_chunk_noise;

pub const CHUNK_SIZE_I: i32 = 32;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE_I * CHUNK_SIZE_I * CHUNK_SIZE_I) as usize;
pub const CHUNK_SIZE_I_PADDED: i32 = CHUNK_SIZE_I + 2;
pub const CHUNK_SIZE_U_PADDED: u32 = CHUNK_SIZE_I_PADDED as u32;

pub struct VoxelWorld {
    pub chunks: Arc<DashMap<IVec3, Chunk34>>,
    pub render_queue: Arc<DashMap<IVec3, ChunkMesh>>,
    pub radius_h: usize,
    pub radius_v: usize,
    mesh_thread_pool: Arc<ThreadPool>,
    mesh_queue: ArrayQueue<IVec3>,
    meshing_queue: Arc<SegQueue<IVec3>>,
    noise: Arc<HybridMulti<Perlin>>,
}

impl VoxelWorld {
    pub fn new(radius_h: usize, radius_v: usize) -> Self {
        let mut noise = HybridMulti::<Perlin>::new(1234);
        noise.octaves = 5;
        noise.frequency = 1.1;
        noise.lacunarity = 2.8;
        noise.persistence = 0.4;
        Self {
            chunks: Arc::new(DashMap::new()),
            render_queue: Arc::new(DashMap::new()),
            mesh_thread_pool: Arc::new(ThreadPoolBuilder::new().num_threads(8).build().unwrap()),
            radius_h,
            radius_v,
            mesh_queue: ArrayQueue::new(2 * radius_h * 2 * radius_h * 2 * radius_v),
            meshing_queue: Arc::new(SegQueue::new()),
            noise: Arc::new(noise),
        }
    }

    pub fn add_change(&mut self, wv: &IVec3, new_block: Block) -> Block {
        let chunk_v = wv_to_cv(wv);
        godot_print!("Adding change to chunk_v={chunk_v}");
        if !self.chunks.contains_key(&chunk_v) {
            self.chunks.insert(chunk_v, Chunk34::empty());
        }

        let mut chunk = self.chunks.get_mut(&chunk_v).unwrap();
        let old_block = chunk.add_change(wv, new_block);
        self.mesh_queue.force_push(chunk_v);

        old_block
    }

    pub fn reset_mesh(&mut self) {
        self.render_queue.clear();
        self.chunks.alter_all(|_, mut chunk| {
            chunk.pipeline = Pipeline::NeedsMesh;
            chunk
        });
    }

    pub fn remove_generated(&mut self) {
        self.render_queue.clear();
        self.chunks.alter_all(|_, mut chunk| {
            chunk.voxels = Chunk34::empty().voxels;
            chunk.generated = false;
            if chunk.changes.len() > 0 {
                chunk.pipeline = Pipeline::NeedsMesh;
            }

            chunk
        });
    }

    // pub fn queue_chunks_to_mesh(&mut self, camera_pos: &Vec3, camera_dir: &Vec3) {
    //     let camera_pos_i = camera_pos.as_ivec3();
    //     let camera_chunk_v = wv_to_chunk_v(&camera_pos_i);
    //     let camera_dir = camera_dir.normalize();

    //     let radius_h = self.radius_h as i32;
    //     let radius_v = self.radius_v as i32;

    //     let mut chunk_vs = Vec::with_capacity(self.radius_h * self.radius_h * self.radius_v);

    //     for x in -radius_h..radius_h {
    //         for y in -radius_v..radius_v {
    //             for z in -radius_h..radius_h {
    //                 let wv = IVec3 {
    //                     x: camera_pos_i.x + (x * CHUNK_SIZE_I),
    //                     y: camera_pos_i.y + (y * CHUNK_SIZE_I),
    //                     z: camera_pos_i.z + (z * CHUNK_SIZE_I),
    //                 };

    //                 let chunk_v = wv_to_chunk_v(&wv);
    //                 chunk_vs.push(chunk_v);
    //             }
    //         }
    //     }

    //     chunk_vs.sort_by(|a, b| {
    //         let a_center = Vec3::from((*a * CHUNK_SIZE_I).as_vec3());
    //         let b_center = Vec3::from((*b * CHUNK_SIZE_I).as_vec3());

    //         let a_vec = a_center - *camera_pos;
    //         let b_vec = b_center - *camera_pos;

    //         let a_score = a_vec.length_squared() - camera_dir.dot(a_vec.normalize()) * 10000.0;
    //         let b_score = b_vec.length_squared() - camera_dir.dot(b_vec.normalize()) * 10000.0;

    //         a_score
    //             .partial_cmp(&b_score)
    //             .unwrap_or(std::cmp::Ordering::Equal)
    //     });

    //     for chunk in chunk_vs {
    //         self.mesh_queue.force_push(chunk);
    //     }
    // }

    pub fn queue_chunks_to_mesh(&mut self, camera_pos: &Vec3, camera_dir: &Vec3) {
        let camera_pos_i = camera_pos.as_ivec3();
        let camera_chunk_v = wv_to_cv(&camera_pos_i);

        let radius_h = self.radius_h as i32;
        let radius_v = self.radius_v as i32;

        let mut chunk_vs = Vec::with_capacity(self.radius_h * self.radius_h * self.radius_v);

        for x in -radius_h..radius_h {
            for y in -radius_v..radius_v {
                for z in -radius_h..radius_h {
                    let wv = IVec3 {
                        x: camera_pos_i.x + (x * CHUNK_SIZE_I),
                        y: camera_pos_i.y + (y * CHUNK_SIZE_I),
                        z: camera_pos_i.z + (z * CHUNK_SIZE_I),
                    };

                    let chunk_v = wv_to_cv(&wv);
                    chunk_vs.push(chunk_v);
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

        let meshing_count = self.meshing_queue.len();

        let available_tasks = self
            .mesh_thread_pool
            .current_num_threads()
            .saturating_sub(meshing_count);

        // if available_tasks == 0 {
        //     return;
        // }

        let mut chunks_to_mesh = vec![];
        for _ in 0..self.mesh_queue.len() {
            // if chunks_to_mesh.len() >= 4 * available_tasks {
            //     break;
            // }
            if let Some(chunk_v) = self.mesh_queue.pop() {
                if !self.chunks.contains_key(&chunk_v) {
                    let mut new_chunk = Chunk34::empty();
                    new_chunk.pipeline = Pipeline::NeedsMesh;
                    self.chunks.insert(chunk_v, new_chunk);
                }

                if self.chunks.get(&chunk_v).unwrap().pipeline == Pipeline::NeedsMesh {
                    chunks_to_mesh.push(chunk_v);
                }
            }
        }

        for chunk_v in &chunks_to_mesh {
            let mut chunk = self.chunks.get_mut(chunk_v).unwrap();
            chunk.pipeline = Pipeline::Meshing;
            self.meshing_queue.push(*chunk_v);
        }

        // if chunks_to_mesh.len() > 0 {
        //     godot_print!("MeshingQueue: {}", chunks_to_mesh.len());
        // }

        let render_queue = self.render_queue.clone();
        let chunks = self.chunks.clone();
        let pool = self.mesh_thread_pool.clone();
        let meshing_queue = self.meshing_queue.clone();
        let noise = self.noise.clone();

        godot::task::spawn(async move {
            for chunk_v in chunks_to_mesh {
                let chunks = chunks.clone();
                let render_queue = render_queue.clone();
                let meshing_queue = meshing_queue.clone();
                let noise = noise.clone();

                pool.spawn(move || {
                    let t1 = std::time::Instant::now();

                    let mut logs = vec![];

                    let clone_time_1 = std::time::Instant::now();

                    let mut working_chunk = chunks.get(&chunk_v).unwrap().clone();

                    logs.push(format!(
                        "[thread] (clonetime_1) - t={:?}",
                        clone_time_1.elapsed()
                    ));

                    if procedural && !working_chunk.generated {
                        working_chunk.solid_count = 0;
                        let chunk_wv = cv_to_wv(&chunk_v);
                        let mut noise_buffer = vec![f64::NAN; Chunk34ShapeI::SIZE as usize];
                        let gen_time = std::time::Instant::now();

                        let gen_log = generate_chunk_noise(
                            &chunk_wv,
                            &working_chunk.shape_i(),
                            4,
                            1000.0,
                            &noise,
                            &mut noise_buffer,
                        );

                        logs.push(gen_log);

                        let terrain_det = std::time::Instant::now();

                        for z in 0..CHUNK_SIZE_I_PADDED {
                            for y in 0..CHUNK_SIZE_I_PADDED {
                                for x in 0..CHUNK_SIZE_I_PADDED {
                                    let lv = IVec3::new(x, y, z);
                                    let wv = chunk_wv + lv - 1; // we are in padded space
                                    let idx = working_chunk.linearize([
                                        lv.x as u32,
                                        lv.y as u32,
                                        lv.z as u32,
                                    ]);
                                    let noise_value = noise_buffer[idx] * 100.0;

                                    if wv.y < -20 {
                                        working_chunk.add_block(idx, Block::Dirt);
                                        continue;
                                    }

                                    // If y is less than the noise sample, we will set the voxel to solid
                                    if (wv.y as f64) < noise_value {
                                        if wv.y >= 120 {
                                            working_chunk.add_block(idx, Block::Snow);
                                        } else if wv.y >= 60 {
                                            working_chunk.add_block(idx, Block::Stone);
                                        } else {
                                            working_chunk.add_block(idx, Block::Grass);
                                        }
                                    } else {
                                        // working_chunk.add_block(idx, Block::None);
                                    }
                                }
                            }
                        }

                        working_chunk.generated = true;

                        logs.push(format!(
                            "[thread] (terrain_det) - t={:?}",
                            terrain_det.elapsed()
                        ));
                        logs.push(format!("[thread] (gen_time) - t={:?}", gen_time.elapsed()));
                    }

                    working_chunk.merge_changes();

                    let clone_time_2 = std::time::Instant::now();

                    let chunk = working_chunk.clone();
                    chunks.insert(chunk_v, chunk);

                    logs.push(format!(
                        "[thread] (clone_time_2) - t={:?}",
                        clone_time_2.elapsed()
                    ));

                    let merge_time = std::time::Instant::now();

                    logs.push(format!(
                        "[thread] (merge_time) - t={:?}",
                        merge_time.elapsed()
                    ));

                    if working_chunk.is_empty() {
                        meshing_queue.pop();
                        return;
                    }

                    let greedy_time = std::time::Instant::now();
                    let mut buffer = GreedyQuadsBuffer::new(CHUNK_VOLUME * 6);
                    greedy_quads(
                        &working_chunk.voxels,
                        &working_chunk.shape_u(),
                        [0; 3],
                        [CHUNK_SIZE_U_PADDED - 1; 3],
                        &RIGHT_HANDED_Y_UP_CONFIG.faces,
                        &mut buffer,
                    );

                    logs.push(format!(
                        "[thread] (greedy_time) - t={:?}",
                        greedy_time.elapsed()
                    ));

                    let mesh_time = Instant::now();

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
                            &working_chunk,
                            &RIGHT_HANDED_Y_UP_CONFIG,
                            &chunk_offset,
                        )
                    };

                    logs.push(format!(
                        "[thread] (mesh_time) - t={:?}",
                        mesh_time.elapsed()
                    ));

                    render_queue.insert(chunk_v, mesh);
                    meshing_queue.pop();

                    logs.push(format!("[thread] (total) - t={:?}", t1.elapsed()));

                    if false {
                        let mut file = OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open("log.txt")
                            .unwrap();

                        writeln!(file, "{}", logs.join("\n")).unwrap();
                    }
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

    pub fn solid(&self) -> u32 {
        match self {
            Block::None => 0,
            _ => 1,
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
    NeedsMesh,
    Meshing,
    Meshed,
    Rendered,
}

pub type Chunk34ShapeI =
    ConstShape3i32<CHUNK_SIZE_I_PADDED, CHUNK_SIZE_I_PADDED, CHUNK_SIZE_I_PADDED>;
pub type Chunk34ShapeU =
    ConstShape3u32<CHUNK_SIZE_U_PADDED, CHUNK_SIZE_U_PADDED, CHUNK_SIZE_U_PADDED>;

#[derive(Clone)]
pub struct Chunk34 {
    voxels: [Block; Chunk34ShapeI::USIZE],
    changes: HashMap<IVec3, Block>,
    generated: bool,
    pub pipeline: Pipeline,
    solid_count: u32,
}

impl Chunk34 {
    fn empty() -> Self {
        Self {
            voxels: [Block::None; Chunk34ShapeI::USIZE],
            changes: HashMap::new(),
            generated: false,
            pipeline: Pipeline::None,
            solid_count: 0,
        }
    }

    pub fn add_change(&mut self, wv: &IVec3, new: Block) -> Block {
        let voxel_v = wv_to_lv(wv);
        self.changes.insert(voxel_v, new);
        let idx = self.linearize_lv(&voxel_v);
        let old_block = self
            .changes
            .insert(voxel_v, new)
            .unwrap_or_else(|| self.voxels[idx]);

        self.pipeline = Pipeline::NeedsMesh;

        old_block
    }

    pub fn add_block(&mut self, idx: usize, new: Block) {
        self.voxels[idx] = new;
        self.solid_count += new.solid();
    }

    pub fn replace_block(&mut self, idx: usize, new: Block) {
        let old = std::mem::replace(&mut self.voxels[idx], new);
        self.solid_count = self.solid_count - old.solid() + new.solid();
    }

    pub fn merge_changes(&mut self) {
        let mutations = self
            .changes
            .iter()
            .map(|(wv, b)| (self.linearize_lv(&wv_to_lv(&wv)), *b))
            .collect::<Vec<(usize, Block)>>();

        for (idx, block) in mutations {
            self.replace_block(idx, block);
        }
    }

    pub fn linearize_lv(&self, p: &IVec3) -> usize {
        Chunk34ShapeI::linearize(p.to_array().map(|s| (s + 1) as i32)) as usize
    }

    pub fn linearize(&self, p: [u32; 3]) -> usize {
        Chunk34ShapeU::linearize(p) as usize
    }

    pub fn shape_i(&self) -> Chunk34ShapeI {
        Chunk34ShapeI {}
    }

    pub fn shape_u(&self) -> Chunk34ShapeU {
        Chunk34ShapeU {}
    }

    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }

    pub fn is_full(&self) -> bool {
        self.solid_count == (CHUNK_SIZE_I * CHUNK_SIZE_I * CHUNK_SIZE_I) as u32
    }
}

pub fn wv_to_cv(wv: &IVec3) -> IVec3 {
    wv.div_euclid(IVec3::splat(CHUNK_SIZE_I))
}

fn cv_to_wv(chunk_v: &IVec3) -> IVec3 {
    chunk_v * CHUNK_SIZE_I
}

fn wv_to_lv(wv: &IVec3) -> IVec3 {
    wv.rem_euclid(IVec3::splat(CHUNK_SIZE_I))
}

#[derive(Debug)]
pub struct ChunkMesh {
    pub indices: Vec<u32>,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub layers: Vec<f32>,
    pub tangents: Vec<f32>,
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
        let idx = self.vert_to_idx(vert) * 4;
        let w = if bi_tangent_preserves_orientation {
            1.0
        } else {
            -1.0
        };
        self.tangents[idx] = tangent[0];
        self.tangents[idx + 1] = tangent[1];
        self.tangents[idx + 2] = tangent[2];
        self.tangents[idx + 3] = w;
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
    let tangents = vec![0.0; num_vertices * 4];

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
