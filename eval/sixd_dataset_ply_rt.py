import os, sys, yaml, cv2, numpy as np

# DEBUG
DEBUG = False

# Gets all train directories
def find_dirs(data_dir):
	# Creates a list of paths to all train directories
	train_dir_list = []
	models_dir_list = []
	for root, dirs, files in os.walk(data_dir):
		if dirs:
			for d in dirs:
				if d == 'train':
					train_dir_list.append(os.path.join(root, d))

				if d == 'models':
					models_dir_list.append(os.path.join(root, d))


	# Sorts by dataset name
	train_dir_list.sort(key=lambda x: x.split(os.sep)[-2])
	models_dir_list.sort(key=lambda x: x.split(os.sep)[-2])

	# Returns list of all train dir paths
	return (train_dir_list, models_dir_list)

# Finds dirs, gt, and info files
def find_files(train_dir_list):
	# Goes through all train dir
	depth_dir_list_all = []
	rgb_dir_list_all = []
	gt_list_all = []
	info_list_all = []
	rt_dir_list_all = []
	for train_dir in train_dir_list:
		# Lists for depth, rgb, gt.yml, and info.yml paths
		depth_dir_list = []
		rgb_dir_list = []
		gt_list = []
		info_list = []
		rt_dir_list = []

		# Walks train directory
		for root, dirs, files in os.walk(train_dir):
			# Finds depth dirs
			if dirs:
				for d in dirs:
					if d == 'rgb':
						# Adds depth path
						p = os.path.join(root, 'depth')
						depth_dir_list.append(p)

						# Adds rgb path
						p = os.path.join(root, 'rgb')
						rgb_dir_list.append(p)

						# Adds gt.yml
						p = os.path.join(root, 'gt.yml')
						gt_list.append(p)

						# Adds info.yml
						p = os.path.join(root, 'info.yml')
						info_list.append(p)

						# Adds rt path
						p = os.path.join(root, 'rt')
						if not os.path.exists(p):
							os.makedirs(p)
						rt_dir_list.append(p)

						# Break
						break
		
		# Sorts stable by object number/id
		depth_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		rgb_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		gt_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		info_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		rt_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))

		# Adds to all lists
		depth_dir_list_all += depth_dir_list
		rgb_dir_list_all += rgb_dir_list
		gt_list_all += gt_list
		info_list_all += info_list
		rt_dir_list_all += rt_dir_list


	# Returns lists of directories
	return (depth_dir_list_all, rgb_dir_list_all, gt_list_all, info_list_all, rt_dir_list_all)

def read_ply_model(ply_path):
	# Opens model ply
	vertex_list = []
	header_list = []
	with open(ply_path) as pf:
		# Goes through header
		vertex_count = 0
		for line in pf:
			# Strips line endings
			l = line.strip(os.linesep)
			header_list.append(l)
			if DEBUG: print(l)

			# If end of header
			if l.startswith('end_header'):
				break

			# Gets vertex count
			elif l.startswith('element vertex'):
				vertex_count = int(l.split(' ')[-1])
				if DEBUG: print(f'vertex_count = {vertex_count}')

		# Goes through vertices
		l = str()
		for count, line in enumerate(pf):
			# Checks if less than num vertices
			if not count < vertex_count:
				if DEBUG: print(f'{count}: {l}')
				if l: line_list = [float(v) for v in l.split(' ')[:-3]] + [int(v) for v in l.split(' ')[-3:]]
				if DEBUG: print(f'{count}: {line_list}')
				break

			# Strips line sep and any whitespace
			l = line.strip(os.linesep).strip(' ')
			if DEBUG and count % 1000 == 0: print(f'{count}: {l}')

			# Copies vertices to list
			# try:
			line_list = [float(v) for v in l.split(' ')]
			# except Exception as e:
			# 	print(f'Exception: {e}')
			# 	print(f'ply_path = {ply_path}')
			# 	print(f"l = {l}")
			# 	print(f"l.split = {l.split(' ')}")
			# 	sys.exit(1)

			if DEBUG and count % 1000 == 0: print(f'{count}: {line_list}')

			# Adds to vertex_list
			vertex_list.append(line_list)

		# Converts to numpy
		vertex_array = np.asarray(vertex_list, dtype=np.float16)
		if DEBUG: print(f'vertex_array.shape = {vertex_array.shape}')

		# Returns numpy array
		return header_list, vertex_array

def rt_model(vertex_array, R, T):
	# Goes through all points
	vector_list = []
	for vertex in vertex_array:
		xyz, rgb = vertex[:3].reshape((3,1)), vertex[-3:].tolist()
		xyz = np.matmul(R, xyz) + T
		xyz = xyz.flatten().tolist()
		vector_list.append(xyz + rgb)
	return np.asarray(vector_list)

def write_ply_model(vertex_array, ply_path):
	# Writes ply
	with open(ply_path, 'w') as pf:
		# Writes header
		pf.write(
			'ply\n' +
			'format ascii 1.0\n' +
			f'element vertex {len(vertex_array)}\n' +
			'property float x\n' +
			'property float y\n' +
			'property float z\n' +
			'property uchar blue\n' +
			'property uchar green\n' +
			'property uchar red\n' + 
			'property uchar alpha\n' +
			'end_header\n'
		)

		# Writes vertices
		for vertex in vertex_array:
			v = [str(v) for v in vertex[:3]] + [str(int(v)) for v in vertex[-3:]] + [str(0)]
			line = ' '.join(v) + '\n'
			pf.write(line)

def make_models_dict(models_dir_list):
	models_dict = {}
	for model_dir in models_dir_list:
		# Gets dataset name and adds to dict
		dataset_name = model_dir.split(os.sep)[-2]
		if dataset_name not in models_dict.keys():
			models_dict.update({dataset_name : {}})

		# Gets model ply and sorts by object number
		for file in os.listdir(model_dir):
			# Creates path to file
			path = os.path.join(model_dir, file)

			# If object ply file
			if file.startswith('obj'):
				# Gets object number
				name, ext = file.rsplit('.', 1)
				obj_number = int(name.split('_')[-1])

				# Adds path to dict
				models_dict[dataset_name].update({obj_number : path})

			# If model info file
			if file.startswith('models_info'):
				models_dict[dataset_name].update({'models_info' : path})

	return models_dict

# Main
def main():
	# Args
	data_dir = sys.argv[1]

	# Gets train and models dirs
	train_dir_list, models_dir_list = find_dirs(data_dir)
	print('Found directories.')

	# Gets the rest of dirs and files we need from train dirs
	depth_dir_list, rgb_dir_list, gt_list, info_list, rt_dir_list = find_files(train_dir_list)
	print('Found files')

	# Create object model ply dictionary
	models_dict = make_models_dict(models_dir_list)
	
	# Converts all in gt
	model_path_prev = None
	for gt_path, rt_dir in zip(gt_list, rt_dir_list):
		# Display
		print(f'Starting: {gt_path}')

		# Splits gt_file path
		gt_split = gt_path.split(os.sep)

		# Gets dataset name
		dataset_name = gt_split[-4]

		# Opens ground truth
		with open(gt_path) as gtf:
			gt_dict = yaml.load(gtf)

			# Goes through gt dict
			for frame_num, frame_list in gt_dict.items():
				# Gets dict
				frame_dict = frame_list[0]

				# Opens model ply if not open
				obj_number = int(frame_dict['obj_id'])
				model_path = models_dict[dataset_name][obj_number]

				# Opens model unless already open
				if not model_path == model_path_prev:
					header_list, vertex_array = read_ply_model(model_path)
					model_path_prev = model_path

				# Gets R & T
				R = np.asarray(frame_dict['cam_R_m2c']).reshape((3,3))
				T = np.asarray(frame_dict['cam_t_m2c']).reshape((3,1))

				# Gets path, converts, and saves
				rt_path = os.path.join(rt_dir, str(frame_num).zfill(4) + '.ply')
				if os.path.exists(rt_path):
					print(f'Already Exists, Skipping... {rt_path}')
					continue
				else:
					rt_to_save = rt_model(vertex_array, R, T)
					write_ply_model(rt_to_save, rt_path)

				# Displays status
				print(f'{rt_path}: Complete')

		# Display
		print(f'Completed: {gt_path}\n')

if __name__ == '__main__':
	main()