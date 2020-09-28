import argparse
import lpips


def compute_perceptual_similarity(path0, path1):
	## Initializing the model
	loss_fn = lpips.LPIPS(net='alex',version='0.1')


	# Load images
	img0 = lpips.im2tensor(lpips.load_image(path0)) # RGB image from [-1,1]
	img1 = lpips.im2tensor(lpips.load_image(path1))


	# Compute distance
	dist01 = loss_fn.forward(img0,img1)
	return float(dist01)