import sys
import argparse
import cv2
import torch
import time
import os
import logging
from tqdm import tqdm

from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def init_models(args):
    # model for face cropping
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640, 640))

    # main model for generation
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')))
    G = G.cuda().half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc = netArc.cuda()
    netArc.eval()

    # model to get face landmarks 
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution
    model = None
    if args.use_sr:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        model = Pix2PixModel(opt)
        model.netG.eval()  # Ensure the model is in evaluation mode

    return app, G, netArc, handler, model


def load_faces(paths, app, crop_size):
    faces = []
    for path in paths:
        try:
            img = cv2.imread(path)
            cropped_faces = crop_face(img, app, crop_size)
            if not cropped_faces:
                raise ValueError(f"No face detected in {path}")
            faces.append(cropped_faces[0][:, :, ::-1])  # First face detected
        except Exception as e:
            logging.error(f"Error loading face from {path}: {e}")
    return faces


def process_source_and_target(args, app):
    # Load and process source images
    logging.info("Processing source images...")
    source = load_faces(args.source_paths, app, args.crop_size)

    # Load and process target images (either from file or video frames)
    set_target = True
    target = []
    if not args.target_faces_paths:
        logging.info("No target faces provided, selecting faces from the video...")
        full_frames, _ = read_video(args.target_video)
        target = get_target(full_frames, app, args.crop_size)
        set_target = False
    else:
        logging.info("Processing target face images...")
        target = load_faces(args.target_faces_paths, app, args.crop_size)

    return source, target, set_target


def perform_inference(source, target, full_frames, app, G, netArc, args):
    # Run inference for face swapping
    with torch.no_grad():
        final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
            full_frames, source, target, netArc, G, app, 
            set_target=True, similarity_th=args.similarity_th, 
            crop_size=args.crop_size, BS=args.batch_size
        )
    return final_frames_list, crop_frames_list, full_frames, tfm_array_list


def enhance_faces(final_frames_list, model):
    # Enhance faces with super resolution if enabled
    logging.info("Enhancing faces using super resolution...")
    return face_enhancement(final_frames_list, model)


def save_output(final_frames_list, crop_frames_list, full_frames, tfm_array_list, args):
    if not args.image_to_image:
        logging.info(f"Saving output video to {args.out_video_name}...")
        get_final_video(final_frames_list, crop_frames_list, full_frames, tfm_array_list, args.out_video_name, fps, handler)
        add_audio_from_another_video(args.target_video, args.out_video_name, "audio")
    else:
        logging.info(f"Saving output image to {args.out_image_name}...")
        result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
        cv2.imwrite(args.out_image_name, result)

    logging.info("Output saved successfully.")


def main(args):
    # Initialize models and parameters
    app, G, netArc, handler, model = init_models(args)

    # Process source and target images
    source, target, set_target = process_source_and_target(args, app)

    # Get full frames from video or load target image
    start_time = time.time()
    logging.info("Starting face swapping...")

    # Perform inference
    full_frames, fps = read_video(args.target_video) if not args.image_to_image else [cv2.imread(args.target_image)]
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = perform_inference(
        source, target, full_frames, app, G, netArc, args
    )

    # Enhance faces if required
    if args.use_sr:
        final_frames_list = enhance_faces(final_frames_list, model)

    # Save the output
    save_output(final_frames_list, crop_frames_list, full_frames, tfm_array_list, args)

    logging.info(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generator params
    parser.add_argument('--G_path', default='weights/G_unet_2blocks.pth', type=str, help='Path to weights for G')
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    parser.add_argument('--num_blocks', default=2, type=int, help='Number of AddBlocks at AddResblock')

    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--crop_size', default=224, type=int, help="Don't change this")
    parser.add_argument('--use_sr', default=False, type=bool, help='True for super resolution on swap images')
    parser.add_argument('--similarity_th', default=0.15, type=float, help='Threshold for selecting a face similar to the target')

    parser.add_argument('--source_paths', default=['examples/images/mark.jpg', 'examples/images/elon_musk.jpg'], nargs='+')
    parser.add_argument('--target_faces_paths', default=[], nargs='+', help="List of target face images")

    # Parameters for image to video
    parser.add_argument('--target_video', default='examples/videos/nggyup.mp4', type=str, help="Target video for swapping faces")
    parser.add_argument('--out_video_name', default='examples/results/result.mp4', type=str, help="Output video name")

    # Parameters for image to image
    parser.add_argument('--image_to_image', default=False, type=bool, help='True for image to image swap, False for image to video swap')
    parser.add_argument('--target_image', default='examples/images/beckham.jpg', type=str, help="Target image for swapping faces")
    parser.add_argument('--out_image_name', default='examples/results/result.png', type=str, help="Output image name")

    args = parser.parse_args()
    main(args)
