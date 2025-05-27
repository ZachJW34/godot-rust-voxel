use cgmath::Point3;
use godot::classes::base_material_3d::TextureParam;
use godot::classes::mesh::PrimitiveType;
use godot::classes::{
    ArrayMesh, BoxMesh, ISprite2D, Mesh, MeshInstance3D, ResourceLoader, Sprite2D,
    StandardMaterial3D, Texture2D,
};
use godot::prelude::*;
use voxel::{Block, CHUNK_SIZE, ToMaterial, VoxelWorld};

mod terrain;
mod voxel;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}

#[derive(GodotClass)]
#[class(base=Sprite2D)]
struct Player {
    #[export]
    speed: f64,
    angular_speed: f64,

    base: Base<Sprite2D>,
}

#[godot_api]
impl ISprite2D for Player {
    fn init(base: Base<Sprite2D>) -> Self {
        godot_print!("Hello, world!"); // Prints to the Godot console

        Self {
            speed: 400.0,
            angular_speed: std::f64::consts::PI,
            base,
        }
    }

    fn physics_process(&mut self, delta: f64) {
        // GDScript code:
        //
        // rotation += angular_speed * delta
        // var velocity = Vector2.UP.rotated(rotation) * speed
        // position += velocity * delta

        let radians = (self.angular_speed * delta) as f32;
        self.base_mut().rotate(radians);

        let rotation = self.base().get_rotation();
        let velocity = Vector2::UP.rotated(rotation) * self.speed as f32;
        self.base_mut().translate(velocity * delta as f32);

        // or verbose:
        // let this = self.base_mut();
        // this.set_position(
        //     this.position() + velocity * delta as f32
        // );
    }
}

#[godot_api]
impl Player {
    #[func]
    fn increase_speed(&mut self, amount: f64) {
        self.speed += amount;
        self.base_mut().emit_signal("speed_increased", &[]);
    }

    #[signal]
    fn speed_increased();
}

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct CubeSpawner {
    base: Base<Node3D>,
    voxels: VoxelWorld,
}

fn p(x: usize, y: usize, z: usize) -> Point3<usize> {
    Point3 { x, y, z }
}

impl ToMaterial for Block {
    fn to_material(&self) -> Gd<StandardMaterial3D> {
        let mut material = StandardMaterial3D::new_gd();

        match self {
            Block::Grass => material.set_albedo(Color::from_rgb(0.1, 0.8, 0.1)),
            Block::Stone => material.set_albedo(Color::from_rgb(0.5, 0.5, 0.5)),
            Block::None => material.set_albedo(Color::from_rgba(0.0, 0.0, 0.0, 0.0)),
            Block::Dirt => todo!(),
            Block::Snow => todo!(),
        }

        material
    }
}

const ARRAY_VERTEX: usize = 0;
const ARRAY_NORMAL: usize = 1;
const ARRAY_TANGENT: usize = 2;
const ARRAY_COLOR: usize = 3;
const ARRAY_TEX_UV: usize = 4;
const ARRAY_TEX_UV2: usize = 5;
const ARRAY_CUSTOM0: usize = 6;
const ARRAY_CUSTOM1: usize = 7;
const ARRAY_CUSTOM2: usize = 8;
const ARRAY_CUSTOM3: usize = 9;
const ARRAY_BONES: usize = 10;
const ARRAY_WEIGHTS: usize = 11;
const ARRAY_INDEX: usize = 12;
const ARRAY_MAX: usize = 13;

#[godot_api]
impl INode3D for CubeSpawner {
    fn init(base: Base<Node3D>) -> Self {
        println!("[CubeSpawner] (init)!");
        Self {
            base,
            voxels: VoxelWorld::init(5, 5, 5),
        }
    }

    fn ready(&mut self) {
        println!("[CubeSpawner] (ready) - Finish");
        self.generate_voxels();
        // self.render_individual_voxels();
        self.render_chunked_voxels();
        self.render_chunked_greedy_voxels();
    }
}

#[godot_api]
impl CubeSpawner {
    #[func]
    fn generate_scene_voxels(&mut self) {
        self.voxels.add_block(&p(0, 0, 0), Block::Grass);
        self.voxels.add_block(&p(1, 0, 0), Block::Stone);
        self.voxels.add_block(&p(0, 1, 0), Block::Snow);
        self.voxels.add_block(&p(1, 1, 0), Block::Dirt);

        self.render_chunked_voxels();
    }

    fn generate_voxels(&mut self) {
        let terrain = terrain::generate_terrain(
            self.voxels.x_chunks * CHUNK_SIZE,
            self.voxels.z_chunks * CHUNK_SIZE,
            0.05,
            CHUNK_SIZE as f64,
            1,
        );
        for point in terrain {
            // godot_print!("[CubeSpawner] (ready) - adding block at {point:?}");
            self.voxels.add_block(&point, Block::Grass);
            for y in 0..point.y {
                self.voxels
                    .add_block(&Point3::new(point.x, y, point.z), Block::Dirt);
            }
        }
        // self.voxels.add_block(&p(0, 0, 0), Block::Grass);
        // self.voxels.add_block(&p(1, 0, 0), Block::Stone);
        // self.voxels.add_block(&p(0, 1, 0), Block::Snow);
        // self.voxels.add_block(&p(1, 1, 0), Block::Dirt);
        // self.voxels.add_block(&p(0, 1, 0), Block::Grass);
        // self.voxels.add_block(&p(2, 0, 0), Block::Grass);
        // self.voxels.add_block(&p(0, 2, 0), Block::Grass);
        // self.voxels.add_block(&p(0, 0, 1), Block::Stone);
        // self.voxels.add_block(&p(0, 0, 2), Block::Stone);
    }

    fn render_individual_voxels(&mut self) {
        let mut cubes: Vec<Gd<MeshInstance3D>> = vec![];
        let identity = Transform3D::IDENTITY;

        for (idx, block) in self.voxels.grid.iter().enumerate() {
            if *block == Block::None {
                continue;
            }

            let point = match self.voxels.idx_to_point(idx) {
                Some(p) => p,
                None => continue,
            };
            println!("[CubeSpawner] (ready) - Creating a cube at grid[{idx}] => {point:?}");
            let mut cube = MeshInstance3D::new_alloc();
            let mut box_mesh = BoxMesh::new_gd();
            cube.set_mesh(&box_mesh);

            let material = block.to_material();
            box_mesh.set_material(&material);
            cube.set_transform(identity.translated(Vector3::new(
                point.x as f32,
                point.y as f32,
                point.z as f32,
            )));

            cubes.push(cube);
        }

        for cube in &cubes {
            self.base_mut().add_child(cube);
        }
    }

    fn render_chunked_voxels(&mut self) {
        let atlas_texture_color = ResourceLoader::singleton()
            .load("res://textures/atlas_color.png")
            .unwrap()
            .cast::<Texture2D>();
        // let atlas_texture_normal = ResourceLoader::singleton()
        //     .load("res://textures/atlas_normalmap.png")
        //     .unwrap()
        //     .cast::<Texture2D>();

        let chunks = self.voxels.mesh_chunks();

        for (vertices, indices) in chunks {
            // godot_print!("[VoxelWorld] (ready) - vertices: {:?}", vertices);
            let mut array = VariantArray::new();
            array.resize(ARRAY_MAX, &Variant::nil());

            let positions = PackedVector3Array::from_iter(
                vertices.iter().map(|v| Vector3::from_array(v.position)),
            );
            let normals = PackedVector3Array::from_iter(
                vertices.iter().map(|v| Vector3::from_array(v.normal)),
            );
            let tangents = PackedFloat32Array::from_iter(vertices.iter().flat_map(|v| v.tangent));
            let uvs = PackedVector2Array::from_iter(
                vertices.iter().map(|v| Vector2::from_array(v.tex_coords)),
            );
            let indices = PackedInt32Array::from_iter(indices.iter().map(|i| *i as i32));

            array.set(ARRAY_VERTEX, &positions.to_variant());
            array.set(ARRAY_NORMAL, &normals.to_variant());
            array.set(ARRAY_TANGENT, &tangents.to_variant());
            array.set(ARRAY_TEX_UV, &uvs.to_variant());
            array.set(ARRAY_INDEX, &indices.to_variant());

            let mut material = StandardMaterial3D::new_gd();
            material.set_texture(TextureParam::ALBEDO, &atlas_texture_color);
            // material.set_texture(TextureParam::NORMAL, &atlas_texture_normal);
            // material.set_normal_scale(100.0);

            let mut mesh = ArrayMesh::new_gd();
            mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &array);
            mesh.surface_set_material(0, &material);

            let mut mesh_instance = MeshInstance3D::new_alloc();
            mesh_instance.set_mesh(&mesh);
            self.base_mut().add_child(&mesh_instance);
        }
    }

    fn render_chunked_greedy_voxels(&mut self) {
        self.voxels.mesh_chunks_greedy();
    }
}
