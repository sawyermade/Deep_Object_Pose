import os, sys, yaml, numpy as np, time, cv2, argparse

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
				# If train directory
				if d == 'train':
					train_dir_list.append(os.path.join(root, d))

				# If models directory
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
	kp_list_all = []
	for train_dir in train_dir_list:
		# Lists for depth, rgb, gt.yml, and info.yml paths
		depth_dir_list = []
		rgb_dir_list = []
		gt_list = []
		info_list = []
		rt_dir_list = []
		kp_dir_list = []
		ply_dir_list = []
		kp_list = []

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

						# Adds ply path and creates rt dir if it doesnt exist
						p = os.path.join(root, 'kp.yml')
						kp_list.append(p)

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
		kp_list.sort(key=lambda x: int(x.split(os.sep)[-2]))

		# Adds to all lists
		depth_dir_list_all += depth_dir_list
		rgb_dir_list_all += rgb_dir_list
		gt_list_all += gt_list
		info_list_all += info_list
		rt_dir_list_all += rt_dir_list
		kp_dir_list_all += kp_dir_list
		ply_dir_list_all += ply_dir_list
		kp_list_all += kp_list

	# Returns lists of directories
	return (depth_dir_list_all, rgb_dir_list_all, gt_list_all, info_list_all, rt_dir_list_all, kp_dir_list_all, ply_dir_list_all, kp_list_all)

# Makes a dictionary with all models ply paths
def make_models_dict(models_dir_list):
	# Goes through all models directories
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

	# Returns model dict of paths, dict[dataset_name][obj_number]
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
				if l: line_list = [float(v) for v in l.split(' ')[:3]] + [int(v) for v in l.split(' ')[6:9]]
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
def calc_cuboid(model_info):
	# Gets model info contents
	diameter, min_x, min_y, min_z, size_x, size_y, size_z = [model_info[k] for k in sorted(model_info.keys())]
	max_x, max_y, max_z = min_x + size_x, min_y + size_y, min_z + size_z
	
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

	return keypoint_array

# Does roation and translation on cuboid keypoints
def rt_kp(kp_array, R, T):
	# Goes through all keypoints
	kp_list = []
	for kp in kp_array:
		# Reshapes xyz vector
		xyz = kp.reshape((3,1))

		# Rotates and translates xyz vector and flattens
		xyz = np.matmul(R, xyz) + T
		xyz = xyz.flatten().tolist()
		kp_list.append(xyz)

	# Returns keypoint list as np array
	return np.asarray(kp_list)

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

# Does roation and translation on object model ply
def rt_model(vertex_array, R, T):
	# Goes through all points
	vector_list = []
	for vertex in vertex_array:
		xyz, rgb = vertex[:3].reshape((3,1)), vertex[-3:].tolist()
		xyz = np.matmul(R, xyz) + T
		xyz = xyz.flatten().tolist()
		vector_list.append(xyz + rgb)
	return np.asarray(vector_list)

# Writes new ply model after R & T
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
			'property uchar red\n' + 
			'property uchar green\n' +
			'property uchar blue\n' +
			'property uchar alpha\n' +
			'end_header\n'
		)

		# Writes vertices
		for vertex in vertex_array:
			v = [str(v) for v in vertex[:3]] + [str(int(v)) for v in vertex[-3:]] + [str(0)]
			line = ' '.join(v) + '\n'
			pf.write(line)

	# Returns true if complete
	return True

# Writes ply file
def write_ply(vertex_list, ply_path):
	# Creates ply path
	# img_dir, img_fname = os.path.split(img_path)
	# img_number, img_ext = img_fname.rsplit('.', 1)
	# ply_dir = os.path.join(os.path.split(img_dir)[0], 'ply')
	# ply_path = os.path.join(ply_dir, f'{img_number}.ply')

	# Creates ply dir if !exists
	# if not os.path.exists(ply_dir):
	# 	os.makedirs(ply_dir)

	# Checks if ply file exists
	# if os.path.exists(ply_path):
	# 	return ply_path

	# Writes ply
	with open(ply_path, 'w') as pf:
		# Writes header
		pf.write(
			'ply\n' +
			'format ascii 1.0\n' +
			f'element vertex {len(vertex_list)}\n' +
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
		for vertex in vertex_list:
			v = [str(v) for v in vertex]
			line = ' '.join(v) + '\n'
			pf.write(line)

	# Return ply path
	return ply_path

def keypoints_3D_to_2D(kp_list, cam_K):
	kp_2d_list = []
	for kp in kp_list:
		kp_temp = (kp / kp[-1]).reshape((3,1))
		kp_2d = np.matmul(cam_K, kp_temp)
		x, y = kp_2d[0] * kp_2d[2], kp_2d[1] * kp_2d[2]
		kp_2d_list.append([int(x),int(y)])

	return np.asarray(kp_2d_list)

def depth_to_ply(depth_img_path, rgb_img_path, cam_K, depth_scale):
	# Opens depth img and gets info
	bgr_img = cv2.imread(rgb_img_path, -1)
	depth_img = cv2.imread(depth_img_path, -1)
	height, width = depth_img.shape[:2]
	fx, _, cx, _, fy, cy, _, _, _ = cam_K

	# Goes through all the pixels and adds all !0 vectors
	vertex_list = []
	for y in range(height):
		for x in range(width):
			# Gets RGB as opencv BGR and depth in millimeters
			B, G, R = bgr_img[y, x]
			Z = depth_img[y, x] * depth_scale

			# If Z == 0, fuck it
			if Z != 0:
				X = (x - cx) * Z / fx
				Y = (y - cy) * Z / fy
				alpha = 0
				vertex_list.append((X, Y, Z, B, G, R, alpha))

	return vertex_list

# Main
def main():
	# Args, needs -d /path/to/dataset. the rest are optional, only makes kp.yml by default
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--directory', help='Path to dataset dir', type=str)
	parser.add_argument('-p', '--ply', help='Depth image to ply point cloud?', default=False, action='store_true')
	parser.add_argument('-r', '--rt', help='Rotate and translate ply object model?', default=False, action='store_true')
	parser.add_argument('-k', '--kp', help='Create rotated and translated keypoint ply?', default=False, action='store_true')
	args = parser.parse_args()

	# Checks required args are there
	if not args.directory:
		parser.print_help()
		sys.exit(1)
	else:
		data_dir = args.directory

	# Gets train and models dirs
	train_dir_list, models_dir_list = find_dirs(data_dir)
	print('Found directories.')

	# Gets the rest of dirs and files we need from train dirs
	depth_dir_list, rgb_dir_list, gt_list, info_list, rt_dir_list, kp_dir_list, ply_dir_list, kp_list = find_files(train_dir_list)
	print('Found files')

	# Create object model ply dictionary
	models_dict = make_models_dict(models_dir_list)
	print('Created models dictionary')
	
	# Converts all in gt
	time_start = time.time()
	print('\nStarting conversions...\n')
	for gt_path, kp_dir, rt_dir, ply_dir, info_path, kp_yml_path, depth_dir, rgb_dir in zip(gt_list, kp_dir_list, rt_dir_list, ply_dir_list, info_list, kp_list, depth_dir_list, rgb_dir_list):
		# Display
		print(f'Starting: {gt_path}')

		# Splits gt_file path
		gt_split = gt_path.split(os.sep)

		# Gets dataset name, tudlight has refined directory this takes care of
		dataset_name = gt_split[-4]
		if dataset_name == 'train':
			dataset_name = 'tudlight'

		# Keypoint dictionary
		kp_dict = {}

		# Opens ground truth and goes through frames
		header_list, vertex_array, model_info = [None]*3
		model_path_prev, keypoint_array = [None]*2
		with open(gt_path) as gtf, open(info_path) as inp:
			# Loads gt
			gt_dict = yaml.load(gtf)

			# Loads camera intrinsic info
			info_dict = yaml.load(inp)

			# Goes through gt dict
			for frame_num, frame_list in gt_dict.items():
				# Gets dict, only one element and always zero
				frame_dict = frame_list[0]
				info_frame_dict = info_dict[int(frame_num)]

				# Gets camera instrinsics for frame
				cam_K = np.asarray(info_frame_dict['cam_K']).reshape((3,3))
				depth_scale = info_frame_dict['depth_scale']

				# creates and saves ply from depth map/image
				if args.ply:
					ply_path = os.path.join(ply_dir, str(frame_num).zfill(4) + '.ply')
					if os.path.exists(ply_path):
						print(f'Already Exists, Skipping... {ply_path}')
					else:
						depth_img_path = os.path.join(depth_dir, str(frame_num).zfill(4) + '.png')
						rgb_img_path = os.path.join(rgb_dir, str(frame_num).zfill(4) + '.png')
						ply_points = depth_to_ply(depth_img_path, rgb_img_path, cam_K.flatten(), depth_scale)
						write_ply(ply_points, ply_path)
						print(f'{ply_path}: Complete')

				# Gets opject/model info
				obj_number = int(frame_dict['obj_id'])
				model_path = models_dict[dataset_name][obj_number]
				model_info_path = models_dict[dataset_name]['models_info']

				# Opens model unless already open
				if not model_path == model_path_prev:
					# Opens object model 
					header_list, vertex_array = read_ply_model(model_path)
					model_path_prev = model_path

					# Opens model_info.yml and calcs keypoints
					with open(model_info_path) as info:
						model_info = yaml.load(info)[obj_number]
						keypoint_array = calc_cuboid(model_info)

				# Gets R & T
				R = np.asarray(frame_dict['cam_R_m2c']).reshape((3,3))
				T = np.asarray(frame_dict['cam_t_m2c']).reshape((3,1))

				# Gets path, converts, and saves keypoints rotated/translated
				kp_3d = rt_kp(keypoint_array, R, T)
				if args.kp:
					kp_ply_path = os.path.join(kp_dir, str(frame_num).zfill(4) + '.ply')
					if os.path.exists(kp_ply_path):
						print(f'Already Exists, Skipping... {kp_ply_path}')
					else:
						write_ply_keypoints(kp_3d, kp_ply_path)
						print(f'{kp_ply_path}: Complete')

				# Gets path, converts, and saves rotated/translated model ply
				if args.rt:
					rt_path = os.path.join(rt_dir, str(frame_num).zfill(4) + '.ply')
					if os.path.exists(rt_path):
						print(f'Already Exists, Skipping... {rt_path}')
					else:
						rt_to_save = rt_model(vertex_array, R, T)
						write_ply_model(rt_to_save, rt_path)
						print(f'{rt_path}: Complete')
				
				# Makes keypoint dict
				kp_2d = keypoints_3D_to_2D(kp_3d, cam_K)
				kp_dict.update({
					int(frame_num) : {
						'3d' : kp_3d.tolist(),
						'2d' : kp_2d.tolist()
					}
				})

			# Saves keypoint dict as yml
			if os.path.exists(kp_yml_path):
				print(f'Already Exists, Skipping... {kp_yml_path}')
			else:
				with open(kp_yml_path, 'w') as kpf:
					yaml.dump(kp_dict, kpf)
					print(f'{kp_yml_path}: Complete')

		# Display
		print(f'Completed: {gt_path}\n')

	# Completed
	time_total = int(time.time() - time_start)
	time_seconds = int(time_total % 60)
	time_minutes = int(time_total // 60)
	time_hours = int(time_minutes // 60)
	time_minutes = int(time_minutes % 60)
	print(f'Conversions Completed: {str(time_hours).zfill(2)}:{str(time_minutes).zfill(2)}:{str(time_seconds).zfill(2)}')

if __name__ == '__main__':
	main()