# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:44:55 2018

@author: B51427
"""

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import tensorflow as tf
import scipy.misc
import numpy as np
import cv2
from tensorflow.python.tools import inspect_checkpoint as chkp

def restore_model(sess):
    
    saver = tf.train.import_meta_graph('./saved_training_model/model.ckpt.meta')
    saver.restore(sess,"./saved_training_model/model.ckpt")
    print("Model restored.")
    chkp.print_tensors_in_checkpoint_file(file_name="./saved_training_model/model.ckpt", tensor_name='', all_tensors=False, all_tensor_names=True)
    
    graph = tf.get_default_graph()
    
    input_image = graph.get_operation_by_name('input_image:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    logits = graph.get_tensor_by_name('logits:0')

    return input_image, keep_prob, logits

def process_image(sess, image, image_shape, input_image, keep_prob, logits):
    im_softmax = sess.run([tf.nn.softmax(logits)],
                           {keep_prob: 1.0, input_image: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return np.array(street_im)

def semantic_segmentation(image):
    
    image_shape = (image.shape[0], image.shape[1]) #shape check
    with tf.Session() as sess:
        input_image, keep_prob, logits = restore_model(sess)
        output_image = process_image(sess, image, input_image, image_shape, keep_prob, logits)
        
    return output_image


def video_run():
    video_output_dir = './runs/videos_output'
    video_data_dir = './data/data_videos'

    videos = ['project_video', 'challenge_video', 'harder_challenge_video', 'night_video', 'city_challenge']

    for video in videos:
        clip = VideoFileClip(video_data_dir+'/'+video+'.mp4')
        output = video_output_dir+'/'+video+'_output.mp4'
    
        road_clip = clip.fl_image(semantic_segmentation) #NOTE: this function expects color images!!
        road_clip.write_videofile(output, audio=False)

if __name__=='__main__':
    video_run()