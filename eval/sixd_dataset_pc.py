import os, sys, yaml, cv2, numpy as np

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

# Makes point clouds
def make_clouds(depth_img_list, info_dict_list):
	# Goes through each object
	obj_cloud_list = []
	for obj_img_list, info_dict in zip(depth_img_list, info_dict_list):
		
		# Goes through images
		img_cloud_list = []
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
			img = cv2.imread(img_path, -1)

			# Gets width and height
			height, width = img.shape[:2]

			# Goes through all the pixels
			vector_list = []
			for y in range(height):
				for x in range(width):
					d = img[y, x]
					Z = d * depth_scale
					X = (x - cx) * Z / fx
					Y = (y - cy) * Z / fy
					vector_list.append((X, Y, Z))

			# Adds to image cloud list
			img_cloud_list.append(vector_list)

		# Adss to object cloud list
		obj_cloud_list.append(img_cloud_list)

	return obj_cloud_list

if __name__ == '__main__':
	# Args
	data_dir = sys.argv[1]
	train_dir = os.path.join(data_dir, 'train')
	out_dir = os.path.join(data_dir, 'pc_train')
	cam_path = os.path.join(data_dir, 'camera.yml')

	# Creates out_dir
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# # Gets camera intrinsics
	# cam_intrinisics = yaml.load(open(cam_path))

	# Gets depth dirs in train folder
	depth_dir_list, rgb_dir_list, gt_list, info_list = find_dirs(train_dir)

	# Load info yamls
	info_dict_list = load_info_ymls(info_list)

	# Gets all depth image paths
	depth_img_list = find_depth_imgs(depth_dir_list)

	# Creates point cloud from depth images
	obj_cloud_list = make_clouds(depth_img_list, info_dict_list)