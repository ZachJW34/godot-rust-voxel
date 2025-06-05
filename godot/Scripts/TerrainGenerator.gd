extends RefCounted

class_name TerrainGenerator

class TerrainJob:
	var thread: Thread
	var result: Array = []
	var is_done: bool = false

func _init():
	jobs = []

var jobs: Array = []

func start_job(chunk: Vector3) -> void:
	var job = TerrainJob.new()
	job.thread = Thread.new()
	job.thread.start(Callable(self, "_generate_terrain").bind(chunk, job))
	jobs.append(job)

func _generate_terrain(chunk: Vector3i, job: TerrainJob) -> void:
	# Simulate some terrain generation
	#print("[Terrain Generator] (_generate_terrain) - Starting job")
	var result = _generate_chunk(Vector3i(chunk), 1234, 32)
	job.result = result
	job.is_done = true
	
func _generate_chunk(origin: Vector3i, world_seed: int, chunk_size: int) -> Array:
	#var t1 = Time.get_ticks_usec()
	var noise = FastNoiseLite.new()
	noise.seed = world_seed
	noise.noise_type = FastNoiseLite.TYPE_PERLIN
	noise.frequency = 1.1
	noise.fractal_octaves = 5
	noise.fractal_lacunarity = 2.8
	noise.fractal_gain = 0.4

	var new_chunk := PackedInt32Array()
	new_chunk.resize(chunk_size * chunk_size * chunk_size)
	new_chunk.fill(1)  # Block::None => 0

	var cache := {}

	for z in range(chunk_size):
		for y in range(chunk_size):
			for x in range(chunk_size):
				var idx = x + y * chunk_size + z * chunk_size * chunk_size
				var local_x = x + origin.x
				var local_y = y + origin.y
				var local_z = z + origin.z
				
#
				if local_y < 1:
					new_chunk[idx] = 1  # Block::Dirt => 1
					continue

				var pos_xz = Vector2(local_x, local_z)
				var sample := 0.0
#
				if cache.has(pos_xz):
					sample = cache[pos_xz]
				else:
					sample = noise.get_noise_2d(float(local_x) / 100.0, float(local_z) / 100.0) * 100.0
					cache[pos_xz] = sample
#
				if float(local_y) < sample:
					new_chunk[idx] = 2   # Block::Grass => 2
				else:
					new_chunk[idx] = 0   # Block::None => 0

	return [origin, new_chunk]
	

func get_finished_jobs() -> Array:
	var results = []
	for job in jobs:
		if job.is_done:
			#print("Finished job result: ", job.result)
			jobs.erase(job)  # Clean up finished jobs
			results.append(job.result)
			break  # Only remove one per frame to avoid iterator invalidation
			
	return results
