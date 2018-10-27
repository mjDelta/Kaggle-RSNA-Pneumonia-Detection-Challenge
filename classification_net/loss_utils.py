# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:59:52 2018

@author: Mengji Zhang
"""

import tensorflow as tf

_EPSILON = 1e-7

def weigthed_binary_crossentropy(target,output,positive_weight=0.75):
  negative_weight=1.0-positive_weight
  output=tf.clip_by_value(output,_EPSILON,1.0-_EPSILON)
  output=-positive_weight*target*tf.log(output)-negative_weight*(1.0-target)*tf.log(1.0-output)
  return output
  
  tf.clip
