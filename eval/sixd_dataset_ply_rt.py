import os, sys, yaml, cv2, numpy as np

# DEBUG
DEBUG = True

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

			# Strips line sep
			l = line.strip(os.linesep)
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

def rt(vertex_array, R, T):
	# Goes through all points
	vector_list = []
	for vertex in vertex_array:
		xyz = vertex[:3].reshape((3,1))
		rgb  = vertex[-3:].tolist()
		newxyz = np.matmul(R, xyz) + T
		newxyz = newxyz.flatten()
		newxyz = newxyz.tolist()
		vector_list.append(newxyz + rgb)
	return np.asarray(vector_list)

def write_ply_model(vertex_array, ply_path='./points_only.ply'):
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

def main():
	# Args
	ply_model_path = sys.argv[1]

	# Gets header and vertices
	header_list, vertex_array = read_ply_model(ply_model_path)

	# Test
	r = [1.00000000, 0.00000000, 0.00000000, 0.00000000, -1.00000000, -0.00000000, 0.00000000, 0.00000000, -1.00000000]
	t = [-0.00000000, 0.00000000, 450.00000000]
	R = np.asarray(r).reshape((3,3))
	T = np.asarray(t).reshape((3,1))
	vertex_array_rt = rt(vertex_array, R, T)
	write_ply_model(vertex_array, './points_only.ply')
	write_ply_model(vertex_array_rt, './points_only_rt.ply')

if __name__ == '__main__':
	main()