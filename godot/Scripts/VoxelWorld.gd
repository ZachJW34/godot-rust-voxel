@tool
class_name VoxelWorld
extends Node

var voxels: Voxels

var _wireframe: bool = false
@export var wireframe: bool:
	get():
		return _wireframe
	set(value):
		_wireframe = value
		voxels.wireframe = _wireframe
		voxels.reset_mesh();
		
var _procedural: bool = false
@export var procedural: bool:
	get():
		return _procedural
	set(value):
		_procedural = value
		voxels.procedural = procedural
		voxels.remove_generated();
		
func _init() -> void:
	voxels = Voxels.new()
	wireframe = voxels.wireframe
	if Engine.is_editor_hint():
		procedural = false
	else:
		procedural = true

func _ready() -> void:
	if Engine.is_editor_hint():
		var viewport = EditorInterface.get_editor_viewport_3d()
		var editor_camera = viewport.get_camera_3d();
		var pos = editor_camera.position;
		print(pos)
		#var cube = MeshInstance3D.new()
		#cube.mesh = BoxMesh.new()  # Creates a 1x1x1 cube centered at origin
		#cube.position = Vector3(32,0,0) # Move it to (5, 0, 0)
		#add_child(cube)
		
		for x in range(0, 31):
			var block = (abs(x) %3)  + 1;
			print(block);
			add_block(Vector3(x, 0, 0), block)
			add_block(Vector3(0, x, 0), block)
			add_block(Vector3(0, 0, x), block)
			
		render()
			
	add_child(voxels)
	
func _physics_process(_time) -> void:
	if !Engine.is_editor_hint():
		voxels.render()
		
func add_block(pos: Vector3, block: int):
	voxels.add_block(vector_to_vector_i(pos), block)
	
func queue_chunks_to_mesh(pos: Vector3):
	voxels.queue_chunks_to_mesh(vector_to_vector_i(pos))
	
func render() -> void:
	voxels.render()
	
func vector_to_vector_i(pos: Vector3):
	var floored = pos.floor()
	var vec3i = Vector3i(floored)
	return vec3i
