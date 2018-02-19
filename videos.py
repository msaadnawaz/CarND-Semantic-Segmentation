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

class SemanticSegmentation(object):
    
    def restore_model(self):
        
        saver = tf.train.import_meta_graph('./saved_training_model/model.meta')
        saver.restore(self.sess,tf.train.latest_checkpoint('./saved_training_model/'))
        print("Model restored.")
        
        graph = tf.get_default_graph()
        
    #    input_image = tf.get_collection("input_image")[0]
    #    keep_prob = tf.get_collection("keep_prob")[0]
    #    logits = tf.get_collection("logits")[0]
        
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.logits = graph.get_tensor_by_name('logits:0')    
        self.input_image = graph.get_tensor_by_name('image_input:0')
        
    
        #return input_image, keep_prob, logits
    
    def process_image(self, image):
        image_shape = (160, 576) #shape check    
        image = scipy.misc.imresize(image, image_shape)
        
        im_softmax = self.sess.run([tf.nn.softmax(self.logits)],
                               {self.keep_prob: 1.0, self.input_image: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        return np.array(street_im)    
    
    def video_run(self):
        video_output_dir = './runs/videos_output'
        video_data_dir = './data/data_videos'
    
        videos = ['project_video', 'challenge_video', 'harder_challenge_video', 'night_video', 'city_challenge']
        
        with tf.Session() as self.sess:
            self.restore_model()
            with tf.Graph().as_default():
                for video in videos:
                    path = video_data_dir+'/'+video+'.mp4'
                    clip = VideoFileClip(path)
                    output = video_output_dir+'/'+video+'_output.mp4'
                    
                    road_clip = clip.fl_image(self.process_image) #NOTE: this function expects color images!!
                    road_clip.write_videofile(output, audio=False)

if __name__=='__main__':
    segmenter = SemanticSegmentation()
    segmenter.video_run()