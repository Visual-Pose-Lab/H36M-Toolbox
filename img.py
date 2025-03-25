import numpy as np
import cv2

def get_3rd_point(a, b):
	direct = a - b
	return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
		center, scale, rot, output_size,
		shift=np.array([0, 0], dtype=np.float32), inv=0
):
	center = np.array(center)
	scale = np.array(scale)

	scale_tmp = scale * 200.0
	src_w = scale_tmp[0]
	dst_w = output_size[0]
	dst_h = output_size[1]

	# rot_rad = np.pi * rot / 180

	# src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
	src_dir = np.array([0, (src_w-1) * -0.5], np.float32)
	dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = center + scale_tmp * shift
	src[1, :] = center + src_dir + scale_tmp * shift
	dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
	dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

	src[2:, :] = get_3rd_point(src[0, :], src[1, :])
	dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans


def crop_image(image, center, scale, output_size):
	"""Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
	Args:
		image numpy array of shape (height, width, 3): input image
		bbox tuple of size 4: input bbox (left, upper, right, lower)

	Returns:
		cropped_image numpy array of shape (height, width, 3): resulting cropped image

	"""

	trans = get_affine_transform(center, scale, 0, output_size)
	image = cv2.warpAffine(
		image,
		trans,
		(output_size),
		flags=cv2.INTER_LINEAR)

	return image