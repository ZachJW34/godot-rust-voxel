extends Node3D

@export var move_speed := 5.0
@export var sprint_multiplier := 2.0
@export var mouse_sensitivity := 0.005
@export var pitch_limit := 1.5 # Radians (~85 degrees)

var yaw := 0.0
var pitch := 0.0

@onready var camera: Camera3D = $Camera3D

func _ready():
	var rot = rotation
	yaw = rot.y
	pitch = rot.x
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func _unhandled_input(event):
	if event is InputEventMouseMotion:
		yaw -= event.relative.x * mouse_sensitivity
		pitch -= event.relative.y * mouse_sensitivity
		pitch = clamp(pitch, -pitch_limit, pitch_limit)
		rotation = Vector3(pitch, yaw, 0)
	if event.is_action_pressed("ui_cancel"): # ESC key by default
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)

	#if Input.is_key_pressed(KEY_SPACE):
		#var cube = MeshInstance3D.new()
		#cube.mesh = BoxMesh.new()
		#cube.position = global_position
		#get_tree().current_scene.add_child(cube)

func _physics_process(delta: float) -> void:
	var direction = Vector3.ZERO

	if Input.is_action_pressed("move_forward"):
		direction -= transform.basis.z
	if Input.is_action_pressed("move_backward"):
		direction += transform.basis.z
	if Input.is_action_pressed("move_left"):
		direction -= transform.basis.x
	if Input.is_action_pressed("move_right"):
		direction += transform.basis.x
	if Input.is_action_pressed("move_down"):
		direction -= transform.basis.y
	if Input.is_action_pressed("move_up"):
		direction += transform.basis.y

	direction = direction.normalized()

	var speed = move_speed
	if Input.is_action_pressed("sprint"):
		speed *= sprint_multiplier

	position += direction * speed * delta
