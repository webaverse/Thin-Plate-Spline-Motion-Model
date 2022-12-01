import torch
import cv2
import os

import matplotlib.animation as animation
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import imageio

from demo import load_checkpoints, make_animation
from demo import find_best_frame as _find
from skimage import img_as_ubyte

from flask import Flask, Response, request, send_from_directory


app = Flask(__name__)


@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	predict_mode = 'relative'  # ['standard', 'relative', 'avd']

	# body = request.get_data()
	image = request.files['image']
	video =  request.files['video']

	# source_image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
	source_image = imageio.imread(image)
	#driving_video_arg = request.args.get('dv')
	dataset_name = request.args.get('dataset')

	pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
	if(dataset_name == 'ted'): # for ted, the resolution is 384*384
		pixel = 384

	from werkzeug.utils import secure_filename
	video.save(secure_filename(video.filename))
	#driving_video_arg = './assets/driving.mp4'
	# source_image = imageio.imread(source_image_path)
	# reader = imageio.get_reader(driving_video_arg)
	reader = imageio.get_reader(video.filename)

	source_image = resize(source_image, (pixel, pixel))[..., :3]

	fps = reader.get_meta_data()['fps']
	driving_video = []
	try:
		for im in reader:
			driving_video.append(im)
	except RuntimeError:
		pass
	reader.close()

	driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

	if predict_mode == 'relative' and find_best_frame:
		i = _find(source_image, driving_video, device.type == 'cpu')
		print ('Best frame: ' + str(i))
		driving_forward = driving_video[i:]
		driving_backward = driving_video[:(i+1)][::-1]
		predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)
		predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)
		predictions = predictions_backward[::-1] + predictions_forward[1:]
	else:
		predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)

	# save resulting video
	imageio.mimsave(str(out_path), [img_as_ubyte(frame) for frame in predictions], fps=fps)
	# return out_path
	os.remove(video.filename)

	# response = Response() # TODO return video
	# response.headers['Content-Type'] = 'video/mp4'
	# response.headers['Access-Control-Allow-Origin'] = '*'
	# response.headers['Access-Control-Allow-Headers'] = '*'
	# response.headers['Access-Control-Allow-Methods'] = '*'
	# response.headers['Access-Control-Expose-Headers'] = '*'
	# response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
	# response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
	# response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
	# return response

	return send_from_directory('./', 'output.mp4', as_attachment=True)


if __name__ == '__main__':
	# edit the config
	device = torch.device('cuda:0')
	#dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
	source_image_path = './assets/source.png'
	driving_video_path = './assets/driving.mp4'
	out_path = './output.mp4'
	config_path = 'config/vox-256.yaml'
	checkpoint_path = 'checkpoints/vox.pth.tar'
	predict_mode = 'relative' # ['standard', 'relative', 'avd']
	find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

	inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

	app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
