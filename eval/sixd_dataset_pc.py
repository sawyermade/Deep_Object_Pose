import os, sys, yaml, cv2, numpy as np, pickle

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

	# print(depth_dir_list, rgb_dir_list, gt_list, info_list)
	return [depth_dir_list, rgb_dir_list, gt_list, info_list]

# Loads info ymls
def load_info_ymls(info_list):
	# Opens each info.yml
	info_dict_list = []
	for info_path in info_list:
		with open(info_path) as ip:
			info_dict = yaml.load(ip)
			info_dict_list.append(info_dict)

	return info_dict_list

# Finds all depth images
def find_depth_imgs(depth_dir_list, ext='png'):
	# Goes through depth dirs/objects
	depth_img_list = []
	for depth_dir in depth_dir_list:

		# Gets all the img file paths
		for root, dirs, files in os.walk(depth_dir):
			if files:
				files = [f for f in files if f.endswith(ext)]
				if files:
					files.sort(key=lambda x: int(x.split('.')[0]))
					img_paths = [os.path.join(root, f) for f in files]
					depth_img_list.append(img_paths)

	return depth_img_list

# Writes ply file
def write_ply(vertex_list, img_path):
	# Creates PLY path
	img_dir, img_fname = os.path.split(img_path)
	img_number, img_ext = img_fname.rsplit('.', 1)
	ply_dir = os.path.join(os.path.split(img_dir)[0], 'ply')
	ply_path = os.path.join(ply_dir, f'{img_number}.ply')

	# Creates ply dir if !exist
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
def make_clouds(depth_img_list, info_dict_list):
	
	# Goes through each object
	obj_number = 1
	for obj_img_list, info_dict in zip(depth_img_list, info_dict_list):
		# DEBUG
		print(f'Object #{obj_number} Started...')

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

			# DEBUG
			print(f'{img_path}:={ply_path}; Complete.')
			sys.exit(0)

		# # DEBUG
		print(f'Object #{obj_number} Complete\n')
		obj_number += 1

	# DEBUG
	print('Depth Map to Point Cloud Complete.\n')

def main():
	# Args
	data_dir = sys.argv[1]
	train_dir = os.path.join(data_dir, 'train')

	# Gets depth dirs in train folder
	depth_dir_list, rgb_dir_list, gt_list, info_list = find_dirs(train_dir)

	# Load info yamls
	info_dict_list = load_info_ymls(info_list)

	# Gets all depth image paths
	depth_img_list = find_depth_imgs(depth_dir_list)

	# Creates point cloud from depth images
	obj_cloud_list = make_clouds(depth_img_list, info_dict_list)

if __name__ == '__main__':
	main()