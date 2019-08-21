import os, sys, yaml, numpy as np

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
	kp_dir_list_all = []
	ply_dir_list_all = []
	for train_dir in train_dir_list:
		# Lists for depth, rgb, gt.yml, and info.yml paths
		depth_dir_list = []
		rgb_dir_list = []
		gt_list = []
		info_list = []
		rt_dir_list = []
		kp_dir_list = []
		ply_dir_list = []

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

						# Adds rt path and creates rt dir if it doesnt exist
						p = os.path.join(root, 'rt')
						if not os.path.exists(p):
							os.makedirs(p)
						rt_dir_list.append(p)

						# Adds kp path and creates rt dir if it doesnt exist
						p = os.path.join(root, 'kp')
						if not os.path.exists(p):
							os.makedirs(p)
						kp_dir_list.append(p)

						# Adds ply path and creates rt dir if it doesnt exist
						p = os.path.join(root, 'ply')
						if not os.path.exists(p):
							os.makedirs(p)
						ply_dir_list.append(p)

						# Break
						break
		
		# Sorts stable by object number/id
		depth_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		rgb_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		gt_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		info_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		rt_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		kp_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
		ply_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))

		# Adds to all lists
		depth_dir_list_all += depth_dir_list
		rgb_dir_list_all += rgb_dir_list
		gt_list_all += gt_list
		info_list_all += info_list
		rt_dir_list_all += rt_dir_list
		kp_dir_list_all += kp_dir_list
		ply_dir_list_all += ply_dir_lis
t
	# Returns lists of directories
	return (depth_dir_list_all, rgb_dir_list_all, gt_list_all, info_list_all, rt_dir_list_all, kp_dir_list_all, ply_dir_list_all)

# Makes a dictionary with all models ply paths
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
			if file.startswith('obj') and file.endswith('.ply'):
				# Gets object number
				name, ext = file.rsplit('.', 1)
				obj_number = int(name.split('_')[-1])

				# Adds path to dict
				models_dict[dataset_name].update({obj_number : path})

			# If model info file
			if file.startswith('models_info'):
				models_dict[dataset_name].update({'models_info' : path})

	return models_dict

# Reads model plys 
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
				if l: line_list = [float(v) for v in l.split(' ')[:3]] + [int(v) for v in l.split(' ')[-3:]]
				if DEBUG: print(f'{count}: {line_list}')
				break

			# Strips line sep and any whitespace
			l = line.strip(os.linesep).strip(' ')
			if DEBUG and count % 1000 == 0: print(f'{count}: {l}')

			# Copies vertices to list
			line_list = [float(v) for v in l.split(' ')]
			if DEBUG and count % 1000 == 0: print(f'{count}: {line_list}')

			# Adds to vertex_list
			vertex_list.append(line_list)

		# Converts to numpy
		vertex_array = np.asarray(vertex_list, dtype=np.float16)
		if DEBUG: print(f'vertex_array.shape = {vertex_array.shape}')

		# Returns numpy array
		return header_list, vertex_array

# Calc cuboid keypoints
def calc_cuboid(model_info, vertex_array):
	# Gets model info contents
	diameter, min_x, min_y, min_z, size_x, size_y, size_z = [model_info[k] for k in sorted(model_info.keys())]
	max_x, max_y, max_z = min_x + size_x, min_y + size_y, min_z + size_z
	# print(min_x, max_x, min_y, max_y, min_z, max_z)
	
	# Create keypoints np array
	keypoint_array = np.zeros((8,3))

	# Top left front/back
	keypoint_array[0] = np.asarray([min_x, min_y, min_z])
	keypoint_array[4] = np.asarray([min_x, min_y, max_z])

	# Top right front/back
	keypoint_array[1] = np.asarray([max_x, min_y, min_z])
	keypoint_array[5] = np.asarray([max_x, min_y, max_z])

	# Bottom left front/back
	keypoint_array[2] = np.asarray([min_x, max_y, min_z])
	keypoint_array[6] = np.asarray([min_x, max_y, max_z])

	# Bottom right front/back
	keypoint_array[3] = np.asarray([max_x, max_y, min_z])
	keypoint_array[7] = np.asarray([max_x, max_y, max_z])

	# print(keypoint_array)
	return keypoint_array

# Writes new keypoint model after R & T
def write_ply_keypoints(keypoint_array, ply_path):
	# Writes ply
	with open(ply_path, 'w') as pf:
		# Writes header
		pf.write(
			'ply\n' +
			'format ascii 1.0\n' +
			f'element vertex {len(keypoint_array)}\n' +
			'property float x\n' +
			'property float y\n' +
			'property float z\n' +
			'property uchar red\n' + 
			'property uchar green\n' +
			'property uchar blue\n' +
			'property uchar alpha\n' +
			'end_header\n'
		)

		# Writes keypoints
		for keypoint in keypoint_array:
			k = [str(k) for k in keypoint] + ['0', '255', '0', '0']
			line = ' '.join(k) + '\n'
			pf.write(line)

	# Returns true if complete
	return True

def test():
	# Paths
	model_path = f'models/obj_{str(obj_number).zfill(2)}.ply'
	model_info = yaml.load(open('models/models_info.yml'))[obj_number]

	# Gets all vertices
	headler_list, vertex_array = read_ply_model(model_path)

	# Calculate cuboid keypoints
	keypoint_array = calc_cuboid(model_info, vertex_array)

	# Write new ply with keypoints
	ret = write_ply_keypoints(vertex_array, keypoint_array, f'./kp_test_{str(obj_number).zfill(2)}.ply')

# Main
def main():
	# Args
	if len(sys.argv) < 2:
		print('Must include path to dataset dir as only argument')
		sys.exit(1)
	data_dir = sys.argv[1]

	# Gets train and models dirs
	train_dir_list, models_dir_list = find_dirs(data_dir)
	print('Found directories.')

	# Gets the rest of dirs and files we need from train dirs
	depth_dir_list, rgb_dir_list, gt_list, info_list, rt_dir_list, kp_dir_list, ply_dir_list = find_files(train_dir_list)
	print('Found files')

	# Create object model ply dictionary
	models_dict = make_models_dict(models_dir_list)
	print('Created models dictionary')
	
	# Converts all in gt
	print('\nStarting conversions...\n')
	model_path_prev = None
	for gt_path, kp_dir in zip(gt_list, kp_dir_list):
		# Display
		print(f'Starting: {gt_path}')

		# Splits gt_file path
		gt_split = gt_path.split(os.sep)

		# Gets dataset name
		dataset_name = gt_split[-4]

		# Opens ground truth
		with open(gt_path) as gtf:
			# Loads gt
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
				kp_path = os.path.join(kp_dir, str(frame_num).zfill(4) + '.ply')
				if os.path.exists(kp_path):
					print(f'Already Exists, Skipping... {kp_path}')
					continue
				else:
					kp_to_save = kp_rt(vertex_array, R, T)
					write_ply_keypoints(kp_to_save, kp_path)

				# Displays status
				print(f'{kp_path}: Complete')

		# Display
		print(f'Completed: {gt_path}\n')

if __name__ == '__main__':
	test()
	# main()