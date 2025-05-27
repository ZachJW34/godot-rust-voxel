@tool
extends Node

func _ready():
	if not Engine.is_editor_hint():
		return # Only run in the editor

	print("[VoxelSceneInit.gd] (_ready) - Initing Scene")
	call_deferred("_find_cube_spawner")

func _find_cube_spawner():
	var root = get_tree().get_edited_scene_root()
	if root:
		var cube_spawner = root.find_child("CubeSpawner", true, false) # Find child by name
		if cube_spawner:
			print("[VoxelSceneInit.gd] (_ready) - Found CubeSpawner: ", cube_spawner)
		else:
			print("[VoxelSceneInit.gd] (_ready) - CubeSpawner not found")
	else:
		print("[VoxelSceneInit.gd] (_ready) - Root not found")
