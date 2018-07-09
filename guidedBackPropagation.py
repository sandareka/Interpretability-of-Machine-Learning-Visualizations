import caffe
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
caffe.set_mode_cpu()
import keras.backend as K
import caffe


def normalize(x):
    return x/(np.sqrt(np.mean(np.square(x)))+1e-5)


def grad_cam(input_model, category_index, layer_name,final_layer,image_size,feature_map_shape,image_path):

    image_name = image_path.split('/')[len(image_path.split('/')) - 1]

    # Make the loss value class specific
    label = np.zeros(input_model.blobs[final_layer].shape)
    label[0, category_index] = 1

    imdiff = input_model.backward(diffs=['data', layer_name], **{input_model.outputs[0]: label})
    gradients = imdiff[layer_name]  # gradients of the loss value/ predicted class score w.r.t conv5 layer

    # Normalizing gradients for better visualization
    gradients = normalize(gradients)
    gradients = gradients[0, :, :, :]

    print("Gradients Calculated")

    activations = input_model.blobs[layer_name].data[0, :, :, :]

    # Calculating importance of each activation map
    weights = np.mean(gradients, axis=(1, 2))

    cam = np.ones(feature_map_shape, dtype=np.float32)

    for i, w in enumerate(weights):
        weighted_activation = w * activations[i, :, :]
        cam = cam + weighted_activation

    # Let's visualize Grad-CAM
    cam = cv2.resize(cam, image_size)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # We are going to overlay the saliency map on the image
    new_image = cv2.imread(image_path)
    new_image = cv2.resize(new_image, image_size)

    cam = np.float32(cam) + np.float32(new_image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # Finally saving the result
    cv2.imwrite("results/"+image_name.split('.')[0] + "_gradCAM.png", cam)

    return heatmap

def deprocess_image(x): #Credits goes to https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py

    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def guided_backprop(net, end, image_path, category_index, final_layer):

    image_name = image_path.split('/')[len(image_path.split('/')) - 1]

    label = np.zeros(net.blobs[final_layer].shape)
    label[0, category_index] = 1

    net.backward(end=end, **{net.outputs[0]: label})

    keys = list(net.blobs.keys())
    nkeys = []
    all_layers_list = list(net.layer_dict)
    for key in keys:
        nkeys.append(key)
        if (key == end):
            break
    for l in reversed(range(len(nkeys) - 1)):
        layer = keys[l]
        next_layer = keys[l + 1]

        if (layer == 'data'):
            imdiff = net.backward(start=next_layer)['data']
        else:
            relu_diff = net.blobs[next_layer].diff
            guidance = np.maximum(relu_diff, 0)
            layer_before_next_layer = all_layers_list[all_layers_list.index(next_layer) - 1]
            if ('relu' not in layer_before_next_layer):
                net.backward(start=next_layer, end=layer)
                continue
            net.blobs[next_layer].diff[...] = guidance
            net.backward(start=next_layer, end=layer)

    gradients = imdiff
    gradients = gradients[0, :, :, :]

    cam = gradients[0, :, :]
    cam = cam + gradients[1, :, :]
    cam = cam + gradients[2, :, :]


    cv2.imwrite("results/" + image_name.split('.')[0] + "_guided_bp.jpg", deprocess_image(cam))
    return gradients

def generate_visualizations():

    # image path
    image_path = 'images/cat_dog.png'

    # image name
    image_name = image_path.split("/")[len(image_path.split("/"))-1]

    # load the model
    net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
                    'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # load input and preprocess it
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # We reshape the image as we classify only one image
    net.blobs['data'].reshape(1, 3, 227, 227)

    # load the image to the data layer of the model
    im = caffe.io.load_image(image_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    # classify the image
    out = net.forward()

    # predicted class
    print(out['fc8'].argmax())

    final_layer = "fc8"  # output layer whose gradients are being calculated
    image_size = (227, 227)  # input image size
    feature_map_shape = (13, 13)  # size of the feature map generated by 'conv5'
    layer_name = 'conv5'  # convolution layer of interest
    category_index = out['fc8'].argmax()  # -if you want to get the saliency map of predicted class or else you can get saliency map for any interested class by specifying here

    heatmap = grad_cam(net, category_index, layer_name, final_layer, image_size, feature_map_shape,image_path)

    guided_gradients = guided_backprop(net, layer_name, image_path, category_index, final_layer)
    new_gradients = np.zeros(shape=(227, 227, 3))
    new_gradients[:, :, 0] = guided_gradients[0, :, :]
    new_gradients[:, :, 1] = guided_gradients[1, :, :]
    new_gradients[:, :, 2] = guided_gradients[2, :, :]

    guided_gradients = np.transpose(guided_gradients, (1, 2, 0))

    guided_gradcam = guided_gradients * heatmap[...,np.newaxis]

    cv2.imwrite("results/" + image_name.split('.')[0] + "_guided_gradcam.jpg", deprocess_image(guided_gradcam))

if __name__ == '__main__':
    generate_visualizations()


