### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import scipy # this is to prevent a potential error caused by importing torch before scipy (happens due to a bad combination of torch & scipy versions)
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from pdb import set_trace as st
from flask import Flask, request, jsonify
from google.cloud import storage
import imageio
import cv2
import uuid
from datetime import datetime

import replicate
from io import BytesIO

#Set the REPLICATE_API_TOKEN environment variable
os.environ["REPLICATE_API_TOKEN"] =  "r8_W9eZaZk2Xdikeun3JeH0GZC97kLuZyY16Kkob"

app = Flask(__name__)

def predict(opt, image_path_list):
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flips

    opt.image_path_list = [image_path_list]
    opt.name = "females_model" 
    opt.which_epoch="latest"
    opt.display_id = 0 
    opt.traverse = True 
    opt.interp_step =  0.15 
    opt.make_video = True
    opt.in_the_wild = True
    opt.verbose = True

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test batches = %d' % (int(dataset_size / len(opt.sort_order))))
    visualizer = Visualizer(opt)
    model = create_model(opt)
    model.eval()

    # create webpage
    if opt.random_seed != -1:
        exp_dir = '%s_%s_seed%s' % (opt.phase, opt.which_epoch, str(opt.random_seed))
    else:
        exp_dir = '%s_%s' % (opt.phase, opt.which_epoch)
    web_dir = os.path.join(opt.results_dir, opt.name, exp_dir)

    if opt.traverse or opt.deploy:
        if opt.traverse:
            out_dirname = 'traversal'
        else:
            out_dirname = 'deploy'
        output_dir = os.path.join(web_dir,out_dirname)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        out_path = ""
        for image_path in opt.image_path_list:
            print(image_path)
            # image_path = "https://storage.googleapis.com/aging-result/0060ccdb-c9d6-473e-865d-2e9d09bf1eac-input.jpg"
            data = dataset.dataset.get_item_from_path(image_path)
            visuals = model.inference(data)
            if opt.traverse and opt.make_video:
                # out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.mp4')
                out_path = "MinhCute.mp4"
                visualizer.make_video(visuals, "MinhCute.mp4")
            elif opt.traverse or (opt.deploy and opt.full_progression):
                if opt.traverse and opt.compare_to_trained_outputs:
                    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_compare_to_{}_jump_{}.png'.format(opt.compare_to_trained_class, opt.trained_class_jump))
                else:
                    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
                visualizer.save_row_image(visuals, out_path, traverse=opt.traverse)
            else:
                out_path = os.path.join(output_dir, os.path.basename(image_path[:-4]))
                visualizer.save_images_deploy(visuals, out_path)

        return out_path
    
    else:
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

        img_path = ''
        # test
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break

            visuals = model.inference(data)
            img_path = data['Paths']
            rem_ind = []
            for i, path in enumerate(img_path):
                if path != '':
                    print('process image... %s' % path)
                else:
                    rem_ind += [i]

            for ind in reversed(rem_ind):
                del img_path[ind]

            visualizer.save_images(webpage, visuals, img_path)

            webpage.save()
            
        return web_dir
    
def save_image_to_gcloud(source_file_name):
    client = storage.Client.from_service_account_json('credentials.json')

    # Set the name of the bucket and the path to the file to upload
    bucket_name = 'aging-output'

    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    destination_blob_name = 'output/' + time_string + str(uuid.uuid4()) + ".gif"

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    # Print the URL of the uploaded file
    print('File uploaded to: {}'.format(blob.public_url))
    return blob.public_url


@app.route('/predict-using-lats-model', methods=['POST'])
def predict_using_lats_model():
    input_image = request.files['file']
    opt = TestOptions().parse(save=False)
    video_output_path = predict(opt, input_image)
    print( "The path of the result is " + video_output_path)

    # video_path = 'input.mp4'  # Path to the input video file
    gif_path = 'output.gif'  # Path to save the output GIF file
    fps = 10  # Frames per second for the output GIF

    # Open the video file
    video = cv2.VideoCapture(video_output_path)

    # Create an empty list to store the frames
    frames = []

    # Read and process each frame from the video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Append the frame to the frames list
        frames.append(frame)

    # Release the video capture
    video.release()

    # Save the frames as a GIF using imageio
    imageio.mimsave(gif_path, frames, fps=fps)

    output_file_path = save_image_to_gcloud(gif_path)

    data = {
        'outputFilePath': output_file_path
    }

    return jsonify(data)

@app.route('/predict-using-sam-model', methods=['POST'])
def predict_using_sam_model():
    print("[START] Predict with SAM model")
    input_image = request.files['file']

    # Đọc dữ liệu từ input_image
    image_data = input_image.read()
   
    output = replicate.run(
        "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
        input={"image": BytesIO(image_data), "target_age": "default"}
    )
    # output_file_path = save_image_to_gcloud(gif_path)

    data = {
        'outputFilePath': output
    }
    print("[END] Predict with SAM model")
    return jsonify(data)

@app.route('/predict-using-sam-model-with-target-age', methods=['POST'])
def predict_using_sam_model_with_target_age():
    print("[START] Predict with SAM model with target age")
    try:
        input_image = request.files['file']
        target_age = int(request.form['targetAge'])

        print('Target age = ' + target_age)
    except (KeyError, ValueError, TypeError):
        # Handle the case when the parameter is missing or not a valid integer
        return "Invalid parameter: 'targetAge' must be provided as an integer", 400

    # Đọc dữ liệu từ input_image
    image_data = input_image.read()
   
    output = replicate.run(
        "yuval-alaluf/sam:9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c",
        input={"image": BytesIO(image_data), "target_age": str(target_age)}
    )
    # output_file_path = save_image_to_gcloud(gif_path)

    data = {
        'outputFilePath': output
    }
    print("[END] Predict with SAM model with target age")
    return jsonify(data)