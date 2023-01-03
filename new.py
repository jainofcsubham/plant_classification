from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, render_template, request
import json
from werkzeug import secure_filename
import os
import numpy as np
import tensorflow as tf


with open('config.json','r') as c:
    params=json.load(c)["params"]
app=Flask(__name__)
app.config['UPLOAD_FOLDER']= params['upload_location']

@app.route("/")
def index():
    return render_template('main2.html')

@app.route("/result", methods=['GET','POST'])
def result():
    if request.method=='POST':
        new=request.files['image']
        file=new.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(new.filename)))
        location_file= os.path.join(app.config['UPLOAD_FOLDER'],new.filename)
        file_name = location_file
        model_file ="./retrained_graph.pb"
        label_file = "./retrained_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "Placeholder"
        output_layer = "final_result"
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        
        with open(model_file, "rb") as f:
             graph_def.ParseFromString(f.read())
        with graph.as_default():
              tf.import_graph_def(graph_def)
              
        t = read_tensor_from_image_file(
                  file_name,
                  input_height=input_height,
                  input_width=input_width,
                  input_mean=input_mean,
                  input_std=input_std)
        
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)
        
        with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
                results = np.squeeze(results)
                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)
        return render_template('result.html', location = new.filename, plant=labels[top_k[0]], prob=results[top_k[0]])        
   
@app.route("/classes")
def classes():
    return render_template('classes.html') 

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph 

app.graph=load_graph('./retrained_graph.pb') 
app.run(debug=True)