import os, sys, cv2, numpy as np, yaml

def view_kp(img, kp_list):
	color_g = (0, 255, 0)
	color_r = (0, 0, 255)
	radius = 10
	for kp in kp_list[:4]:
		cv2.circle(img, tuple(kp), radius, color_g)
	for kp in kp_list[4:]:
		cv2.circle(img, tuple(kp), radius, color_r)

	cv2.imshow('kp', img)
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()

def main():
	# Args for paths
	# img_path = sys.argv[1]
	# kp_yml_path = sys.argv[2]
	img_path = '1000.png'
	kp_yml_path = 'kp.yml'

	# Opens img
	img_bgr = cv2.imread(img_path, -1)

	# Opens kp yml as dict
	with open(kp_yml_path) as kpf:
		kp_dict = yaml.load(kpf)
	# print(kp_dict)

	# Img filename
	img_fname = img_path.split(os.sep)[-1]
	img_num, img_ext = img_fname.rsplit('.', 1)

	# Gets 2d kp list
	kp_list = kp_dict[int(img_num)]['2d']

	# Visualizes kps
	view_kp(img_bgr, kp_list)

if __name__ == '__main__':
	main()