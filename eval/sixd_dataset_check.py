import cv2, os, sys, numpy as np

if __name__ == '__main__':
	bb = [257, 160, 118, 155]
	p1, p2 = (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3])
	color = (0, 255, 0) #Green
	depth_max = 2**16 - 1

	# fdir = '/data/DATASETS/linemod_plus/doumanoglou/train/01/rgb/'
	fdir = '/data/DATASETS/linemod_plus/doumanoglou/train/01/depth/'
	fname = '0500.png'
	fpath = os.path.join(fdir, fname)
	print(f'File path = {fpath}')

	im = cv2.imread(fpath)
	im[im > 0] = depth_max
	cv2.rectangle(im, p1, p2, color)
	print(f'Image shape = {im.shape}')

	cv2.imshow('bb', im)
	k = cv2.waitKey(0)
	if k == 27: cv2.destroyAllWindows()