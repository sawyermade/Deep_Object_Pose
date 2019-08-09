import os, sys, yaml, cv2, numpy as np

DEBUG = False
DEBUG_ONE = False

# Finds dirs, gt, and info files
def find_dirs(train_dir):
	# Lists for depth, rgb, gt.yml, and info.yml paths
	depth_dir_list = []
	rgb_dir_list = []
	gt_list = []
	info_list = []

	# Walks train directory
	for root, dirs, files in os.walk(train_dir):
		# Finds depth dirs
		if dirs:
			for d in dirs:
				if d == 'depth':
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

					# Break
					break
	
	# Sorts by object number/id
	depth_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
	rgb_dir_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
	gt_list.sort(key=lambda x: int(x.split(os.sep)[-2]))
	info_list.sort(key=lambda x: int(x.split(os.sep)[-2]))

	# Returns lists of directories
	return (depth_dir_list, rgb_dir_list, gt_list, info_list)

# Loads info ymls
def load_info_ymls(info_list):
	# Opens each info.yml
	info_dict_list = []
	for info_path in info_list:
		with open(info_path) as ip:
			info_dict = yaml.load(ip)
			info_dict_list.append(info_dict)

	# Returns list to all info.yml for dataset subset
	return info_dict_list

# Finds all depth images
def find_depth_imgs(depth_dir_list, ext='png'):
	# Goes through depth dirs/objects
	depth_img_list = []
	for depth_dir in depth_dir_list:
		# Gets all the img file paths
		for root, dirs, files in os.walk(depth_dir):
			# If there are files
			if files:
				files = [f for f in files if f.endswith(ext)]

				# If there are any .ext files
				if files:
					# Sorts by integer filename, gets full path to files, and adds to list
					files.sort(key=lambda x: int(x.rsplit('.', 1)[0]))
					img_paths = [os.path.join(root, f) for f in files]
					depth_img_list.append(img_paths)

	# Returns list of all depth img paths for dataset subset
	return depth_img_list

# Writes ply file
def write_ply(vertex_list, img_path):
	# Creates ply path
	img_dir, img_fname = os.path.split(img_path)
	img_number, img_ext = img_fname.rsplit('.', 1)
	ply_dir = os.path.join(os.path.split(img_dir)[0], 'ply')
	ply_path = os.path.join(ply_dir, f'{img_number}.ply')

	# Creates ply dir if !exists
	if not os.path.exists(ply_dir):
		os.makedirs(ply_dir)

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

# Makes point clouds
def make_plys(depth_img_list, info_dict_list, verbose=False):
	# VERBOSE
	dataset_name = depth_img_list[0][0].split(os.sep)[-5]
	if verbose:
		print(f'Dataset:{dataset_name} Start')

	# Goes through each object
	obj_number = 1
	for obj_img_list, info_dict in zip(depth_img_list, info_dict_list):
		# VERBOSE
		if verbose: 
			print(f'Dataset:{dataset_name} Object:{str(obj_number).zfill(2)}')

		# Goes through images
		for img_path in obj_img_list:
			# Gets img filename and number
			img_filename = img_path.split(os.sep)[-1]
			img_num = int(img_filename.split('.')[0])

			# Gets intrinsic for img
			frame_dict = info_dict[img_num]
			cam_K = frame_dict['cam_K']
			depth_scale = frame_dict['depth_scale']
			view_level = frame_dict['view_level']

			# Sets intrinsics
			fx, _, cx, _, fy, cy, _, _, _ = cam_K

			# Opens image as is
			img_dir, img_fname = os.path.split(img_path)
			img_rgb_path = os.path.join(os.path.split(img_dir)[0], 'rgb', img_fname)
			img_bgr = cv2.imread(img_rgb_path, -1)
			img_depth = cv2.imread(img_path, -1)

			# Gets width and height
			height, width = img_depth.shape[:2]

			# Goes through all the pixels
			vertex_list = []
			for y in range(height):
				for x in range(width):
					B, G, R = img_bgr[y, x]
					Z = img_depth[y, x] * depth_scale
					if Z == 0: continue
					X = (x - cx) * Z / fx
					Y = (y - cy) * Z / fy
					vertex_list.append((X, Y, Z, B, G, R, 0))

			# Writes ply file
			ply_path = write_ply(vertex_list, img_path)

			# VERBOSE or DEBUG
			if verbose or DEBUG: print(f'{img_path}:={ply_path}')

			# DEBUG_ONE
			if DEBUG_ONE:
				sys.exit(0)

		# VERBOSE
		if verbose: print(f'Dataset:{dataset_name} Object:{obj_number} Complete')
		
		# Object increment
		obj_number += 1

	# VERBOSE
	if verbose: print(f'Dataset:{dataset_name} End')

def main():
	# Verbose var
	verbose = False

	# Checks VERBOSE, if there is second positional argument either true, t, or v
	min_args = 1
	num_args = 2
	if len(sys.argv) > num_args:
		if sys.argv[num_args].lower() == 'true' or sys.argv[num_args].lower() == 't' or sys.argv[num_args].lower() == 'v'  or sys.argv[num_args].lower() == '--verbose' or sys.argv[num_args].lower() == '-v':
			verbose = True

	elif len(sys.argv) == num_args:
		print(f'Optional {num_args - min_args} extra positional arg(s): --verbose')

	elif len(sys.argv) < num_args:
		# Required arg is path to BOP/SIXD dataset directory, optional is weather to be verbose or not
		print(f'Required {min_args} positional arg(s): /path/to/dataset/dir')
		sys.exit(1)

	# Gets all train directories
	data_dir = sys.argv[1] # This is the eith main dataset path ex:/pathTo/sixdds/ with all dataset subsets or path to train dir of single dataset subset ex:/pathTo/sixdds/doumanoglou/train
	train_dir_list = []
	for root, dirs, files in os.walk(data_dir):
		if dirs:
			for d in dirs:
				if d == 'train':
					train_dir_list.append(os.path.join(root, d))

	# Checks if its one dataset subset or not (sys.argv[1] == single train dir)
	if not train_dir_list:
		train_dir_list.append(data_dir)

	# Goes through all training directories
	for train_dir in train_dir_list:
		# Gets depth dirs in train folder
		depth_dir_list, rgb_dir_list, gt_list, info_list = find_dirs(train_dir)
		# VERBOSE
		if verbose: print('Found directories')

		# Load info yamls
		info_dict_list = load_info_ymls(info_list)
		# VERBOSE
		if verbose: print('Loaded infos')

		# Gets all depth image paths
		depth_img_list = find_depth_imgs(depth_dir_list)
		# VERBOSE
		if verbose: print('Found depth img paths')

		# Creates point cloud from depth images
		make_plys(depth_img_list, info_dict_list, verbose)
		# VERBOSE
		if verbose: print(f'Completed ply conversions: {train_dir}\n')

if __name__ == '__main__':
	main()