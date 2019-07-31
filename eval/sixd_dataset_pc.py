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

	return [depth_dir_list, rgb_dir_list, gt_list, info_list]

# Makes point clouds
def make_clouds(depth_img_list, cam_intrinisics):
	# Goes through each object
	for obj_img_list in depth_img_list:
		# Goes through images
		for img_path in obj_img_list:

			# Opens image as is
			img = cv2.imread(img_path, -1)

			# DEBUG
			print(img2.shape)
			print(f'img.shape = {img.shape}')
			# cv2.imshow('depth', img)
			# k = cv2.waitKey(0)
			# if k == 27: cv2.destroyAllWindows()

			# Goes through each pixel
			dmax = 2**16 - 1
			for y in range(img.shape[0]):
				for x in range(img.shape[1]):
					if img[y, x] > 0:
						print(img[y, x])

# Finds all depth images
def find_depth_imgs(depth_dir_list, ext='png'):
	# Goes through depth dirs/objects
	depth_img_list = []
	for d in depth_dir_list:

		# Gets all the img file paths
		for root, dirs, files in os.walk(d):
			if files:
				files = [f for f in files if f.endswith(ext)]
				img_paths = [os.path.join(root, f) for f in files]
				depth_img_list.append(img_paths)

	return depth_img_list

if __name__ == '__main__':
	# Args
	data_dir = sys.argv[1]
	train_dir = os.path.join(data_dir, 'train')
	out_dir = os.path.join(data_dir, 'pc_train')
	cam_path = os.path.join(data_dir, 'camera.yml')

	# Creates out_dir
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Gets camera intrinsics
	cam_intrinisics = yaml.load(open(cam_path))

	# Gets depth dirs in train folder
	depth_dir_list, rgb_dir_list, gt_list, info_list = find_dirs(train_dir)

	# Gets all depth image paths
	depth_img_list = find_depth_imgs(depth_dir_list)

	# Creates point cloud from depth images
	pc_list = make_clouds(depth_img_list, cam_intrinisics)