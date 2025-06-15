use std::collections::HashMap;

use glam::{IVec3, Vec3};
use godot::classes::mesh::{ArrayFormat, PrimitiveType};
use godot::classes::{
    ArrayMesh, CompressedTexture2DArray, MeshInstance3D, ResourceLoader, Shader, ShaderMaterial,
};
use godot::obj::{EngineBitfield, WithBaseField};
use godot::prelude::*;
use voxel::{Block, VoxelWorld, wv_to_cv};

mod terrain;
mod voxel;

struct VoxelExtension;

#[gdextension]
unsafe impl ExtensionLibrary for VoxelExtension {}

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct Voxels {
    base: Base<Node3D>,
    world: VoxelWorld,
    #[var]
    wireframe: bool,
    #[var]
    procedural: bool,
    // wireframe_shader: Gd<Shader>,
    // voxel_shader: Gd<Shader>,
    // texture_array: Gd<CompressedTexture2DArray>,
    main_shader_material: Gd<ShaderMaterial>,
    wireframe_shader_material: Gd<ShaderMaterial>,
    camera_pos: Vec3,
    rendered_chunks: HashMap<IVec3, Gd<Node>>,
}

const ARRAY_VERTEX: usize = 0;
const ARRAY_NORMAL: usize = 1;
const ARRAY_TANGENT: usize = 2;
// const ARRAY_COLOR: usize = 3;
const ARRAY_TEX_UV: usize = 4;
// const ARRAY_TEX_UV2: usize = 5;
const ARRAY_CUSTOM0: usize = 6;
// const ARRAY_CUSTOM1: usize = 7;
// const ARRAY_CUSTOM2: usize = 8;
// const ARRAY_CUSTOM3: usize = 9;
// const ARRAY_BONES: usize = 10;
// const ARRAY_WEIGHTS: usize = 11;
const ARRAY_INDEX: usize = 12;
const ARRAY_MAX: usize = 13;

#[godot_api]
impl INode3D for Voxels {
    fn init(base: Base<Node3D>) -> Self {
        println!("[Voxels] (init)!");
        let texture_array = ResourceLoader::singleton()
            .load("res://textures/texture_array.png")
            .unwrap()
            .cast::<CompressedTexture2DArray>();

        let wireframe_shader = ResourceLoader::singleton()
            .load("res://shaders/wireframe.gdshader")
            .unwrap()
            .cast::<Shader>();

        let main_shader = ResourceLoader::singleton()
            .load("res://shaders/voxel.gdshader")
            .unwrap()
            .cast::<Shader>();

        let mut wireframe_shader_material = ShaderMaterial::new_gd();
        wireframe_shader_material.set_shader(&wireframe_shader);

        let mut main_shader_material = ShaderMaterial::new_gd();
        main_shader_material.set_shader(&main_shader);
        main_shader_material.set_shader_parameter("texture_array", &texture_array.to_variant());

        Self {
            base,
            world: VoxelWorld::new(40, 4),
            wireframe: false,
            procedural: true,
            wireframe_shader_material,
            main_shader_material,
            camera_pos: Vec3::MAX,
            rendered_chunks: HashMap::new(),
        }
    }

    // fn ready(&mut self) {
    //     println!("[CubeSpawner] (ready) - Finish");
    //     self.generate_voxels();
    //     // self.render_individual_voxels();
    //     self.render_chunked_voxels();
    //     self.render_chunked_greedy_voxels();
    // }
}

trait ToGlamVec3 {
    fn as_vec3(&self) -> Vec3;
}

trait ToGlamIVec3 {
    fn as_ivec3(&self) -> IVec3;
}

impl ToGlamVec3 for Vector3 {
    fn as_vec3(&self) -> Vec3 {
        Vec3::from_array(self.to_array())
    }
}

impl ToGlamIVec3 for Vector3 {
    fn as_ivec3(&self) -> IVec3 {
        self.as_vec3().as_ivec3()
    }
}

#[godot_api]
impl Voxels {
    // #[func]
    // fn set_world_params(&mut self) {
    //     self.world = VoxelWorld::init();
    // }

    #[func]
    fn add_block(&mut self, point: Vector3, block: i32) {
        let point = point.as_ivec3();
        self.world.add_change(&point, Block::from(block));
    }

    #[func]
    fn remove_generated(&mut self) {
        self.remove_all_nodes();
        self.world.remove_generated();
    }

    #[func]
    fn reset_mesh(&mut self) {
        self.remove_all_nodes();
        self.world.reset_mesh();
    }

    #[func]
    fn queue_chunks_to_mesh(&mut self, camera_pos: Vector3, camera_dir: Vector3) {
        let camera_pos = camera_pos.as_vec3();
        let camera_pos_i = camera_pos.as_ivec3();
        let camera_dir = camera_dir.as_vec3();

        let chunk_key_old = wv_to_cv(&self.camera_pos.as_ivec3());
        let chunk_key_new = wv_to_cv(&camera_pos_i);

        // Only search for new meshes on chunk change
        if chunk_key_new == chunk_key_old {
            return;
        }

        self.camera_pos = camera_pos;
        self.world.queue_chunks_to_mesh(&camera_pos, &camera_dir);
    }

    fn remove_all_nodes(&mut self) {
        let children = self.base_mut().get_children();
        for node in children.iter_shared() {
            self.base_mut().remove_child(&node);
        }
    }

    #[func]
    fn render(&mut self) {
        let mut timings = vec![];
        timings.push("[VoxelPlugin] (render) - timings:".to_string());
        let t = std::time::Instant::now();

        self.world
            .mesh_queued_chunks(&self.camera_pos.as_ivec3(), self.procedural);

        let shader_material = if self.wireframe {
            // Is clone an issue here?
            &self.wireframe_shader_material.clone()
        } else {
            &self.main_shader_material.clone()
        };

        let mut chunks_to_render: Vec<_> = self
            .world
            .render_queue
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        // if chunks_to_render.len() > 0 {
        //     godot_print!("Rendering {} chunks", chunks_to_render.len());
        // }

        let camera_chunk_v = wv_to_cv(&self.camera_pos.as_ivec3());
        chunks_to_render.sort_by_key(|chunk_v| (camera_chunk_v - chunk_v).length_squared());

        timings.push(format!("\tt_setup={:?}", t.elapsed()));

        let format = ArrayFormat::VERTEX
            | ArrayFormat::NORMAL
            | ArrayFormat::TANGENT
            | ArrayFormat::TEX_UV
            | ArrayFormat::INDEX
            | ArrayFormat::CUSTOM0
            | ArrayFormat::from_ord(4 << ArrayFormat::CUSTOM0_SHIFT.ord());

        // let chunks_to_render: Vec<_> = chunks_to_render.iter().take(25).collect();

        if chunks_to_render.len() > 0 {
            godot_print!("RenderQueue: {}", chunks_to_render.len());
        }

        timings.push(format!("\tloop_start ({} chunks):", chunks_to_render.len()));
        let t = std::time::Instant::now();
        for chunk_v in chunks_to_render {
            let t = std::time::Instant::now();
            let (_, chunk_mesh) = self.world.render_queue.remove(&chunk_v).unwrap();
            if self.rendered_chunks.contains_key(&chunk_v) {
                godot_print!("Removing chunk...");
                let child = self.rendered_chunks.remove(&chunk_v).unwrap();
                self.base_mut().remove_child(&child);
                child.free();
            }

            if chunk_mesh.positions.len() <= 1 {
                continue;
            }

            timings.push(format!("\t\tcheck_chunk={:?}", t.elapsed()));
            let t = std::time::Instant::now();

            let indices = PackedInt32Array::from_iter(chunk_mesh.indices.iter().map(|i| *i as i32));
            let positions = PackedVector3Array::from_iter(
                chunk_mesh.positions.iter().map(|p| Vector3::from_array(*p)),
            );
            let normals = PackedVector3Array::from_iter(
                chunk_mesh.normals.iter().map(|n| Vector3::from_array(*n)),
            );
            let tangents = PackedFloat32Array::from(chunk_mesh.tangents);
            let uvs = PackedVector2Array::from_iter(
                chunk_mesh.uvs.iter().map(|uv| Vector2::from_array(*uv)),
            );
            let layers = PackedFloat32Array::from(chunk_mesh.layers);

            timings.push(format!("\t\tmapping={:?}", t.elapsed()));
            let t = std::time::Instant::now();

            let mut array = VariantArray::new();
            array.resize(ARRAY_MAX, &Variant::nil());

            array.set(ARRAY_VERTEX, &positions.to_variant());
            array.set(ARRAY_NORMAL, &normals.to_variant());
            array.set(ARRAY_TANGENT, &tangents.to_variant());
            array.set(ARRAY_TEX_UV, &uvs.to_variant());
            array.set(ARRAY_INDEX, &indices.to_variant());
            array.set(ARRAY_CUSTOM0, &layers.to_variant());

            let mut mesh = ArrayMesh::new_gd();

            mesh.add_surface_from_arrays_ex(PrimitiveType::TRIANGLES, &array)
                .flags(format)
                .done();

            mesh.surface_set_material(0, shader_material);

            let mut mesh_instance = MeshInstance3D::new_alloc();
            mesh_instance.set_mesh(&mesh);
            mesh_instance.set_name(&chunk_v.to_string());

            self.base_mut().add_child(&mesh_instance);
            mesh_instance.set_owner(&*self.base_mut());

            self.rendered_chunks.insert(chunk_v, mesh_instance.upcast());

            timings.push(format!("\t\tmeshing={:?}", t.elapsed()));
        }

        let mut nodes_to_remove = vec![];

        for chunk_v in self.rendered_chunks.keys() {
            let dis = (camera_chunk_v - chunk_v).abs();
            if dis.x > self.world.radius_h as i32
                || dis.z > self.world.radius_h as i32
                || dis.y > self.world.radius_v as i32
            {
                nodes_to_remove.push(*chunk_v);
            }
        }

        for chunk_v in nodes_to_remove {
            let node = self.rendered_chunks.remove(&chunk_v).unwrap();
            self.base_mut().remove_child(&node);
            node.free();
            self.world.chunks.remove(&chunk_v);
        }

        timings.push(format!("\tloop_end={:?}", t.elapsed()));
        // godot_print!("{}", timings.join("\n"));
    }
}
