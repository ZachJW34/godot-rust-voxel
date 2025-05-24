use cgmath::Point3;
use godot::classes::mesh::PrimitiveType;
use godot::classes::{
    ArrayMesh, BoxMesh, ISprite2D, Mesh, MeshInstance3D, Sprite2D, StandardMaterial3D,
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
            Block::None => material.set_albedo(Color::from_rgba(0.0, 0.0, 0.0, 0.0)), // or skip rendering
        }

        material
    }
}

#[godot_api]
impl INode3D for CubeSpawner {
    fn init(base: Base<Node3D>) -> Self {
        println!("[CubeSpawner] (init)");
        Self {
            base,
            voxels: VoxelWorld::init(),
        }
    }

    fn ready(&mut self) {
        let terrain = terrain::generate_terrain(CHUNK_SIZE, CHUNK_SIZE, 0.05, CHUNK_SIZE as f64, 1);
        for point in terrain {
            self.voxels.add_block(&point, Block::Grass);
        }
        // self.voxels.add_block(&p(0, 0, 0), Block::Grass);
        // self.voxels.add_block(&p(2, 0, 0), Block::Grass);
        // self.voxels.add_block(&p(0, 2, 0), Block::Grass);
        // self.voxels.add_block(&p(0, 0, 2), Block::Stone);

        let mut cubes: Vec<Gd<MeshInstance3D>> = vec![];

        let identity = Transform3D::IDENTITY;

        for (idx, block) in self.voxels.grid.iter().enumerate() {
            if *block == Block::None {
                continue;
            }

            // let mesh = ArrayMesh::new_gd();
            // mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, arrays);

            let point = self.voxels.idx_to_point(idx);
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

        println!("[CubeSpawner] (ready) - Finish");
    }
}
