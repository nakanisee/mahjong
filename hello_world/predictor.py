import numpy as xp
import sys
import cv2
import time
import math
from PIL import Image, ImageDraw, ImageFont

def area_of(left_top, right_bottom):
    hw = xp.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = xp.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = xp.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    class_index = box_scores[:, 5]
    print(class_index)
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]
    picked = []
    #_, indexes = scores.sort(descending=True)
    indexes = xp.argsort(scores)
    #indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        #current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        #indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            xp.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def convert_locations_to_boxes(locations, center_variance, size_variance, image_size):
    h = w = 20 / image_size
    boxes = xp.zeros(locations.shape, dtype = xp.float32)
    print(locations.shape)
    for y in range(locations.shape[2]):
        for x in range(locations.shape[3]):
            boxes[:,0,y,x] = locations[:,0,y,x] * center_variance * w + (x+0.5) * 8/image_size
            boxes[:,1,y,x] = locations[:,1,y,x] * center_variance * h + (y+0.5) * 8/image_size
            boxes[:,2,y,x] = xp.exp(locations[:,2,y,x] * size_variance) * w
            boxes[:,3,y,x] = xp.exp(locations[:,3,y,x] * size_variance) * h
    return boxes

def center_form_to_corner_form(locations):
    return xp.concatenate([locations[..., :2] - locations[..., 2:]/2,
                     locations[..., :2] + locations[..., 2:]/2], len(locations.shape) - 1)

def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def chainer_im2col(
        img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = xp.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = xp.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return col


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = xp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=xp.float32)

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def convolution2d(x, weights, bias=0, stride=1, pad=0):
    FN, FC, FH, FW = weights.shape
    N, C, H, W = x.shape
    out_h = 1 + int((H + 2*pad - FH) / stride)
    out_w = 1 + int((W + 2*pad - FW) / stride)
    col = im2col(x, FH, FW, stride, pad)
    col_w = weights.reshape(FN, -1).T
    out = xp.dot(col, col_w) + bias
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    return out

def convolution2d(x, weights, bias=0, stride=1, pad=0):
    FN, FC, FH, FW = weights.shape
    N, C, H, W = x.shape
    out_h = 1 + int((H + 2*pad - FH) / stride)
    out_w = 1 + int((W + 2*pad - FW) / stride)
    col = im2col(x, FH, FW, stride, pad)
    col_w = weights.reshape(FN, -1).T
    out = xp.dot(col, col_w) + bias
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    return out


def depthwise_convolution2d(x, weights, bias=None, stride=1, pad=0):
    FN, FC, FH, FW = weights.shape
    N, C, H, W = x.shape
    G = C
    N, iC, iH, iW = x.shape
    oC, _, kH, kW = weights.shape  # _ == iCg
    iCg = iC // G
    oCg = oC // G

    # (N, iC, kW, kW, oH, oW)
    x = chainer_im2col(x, kH, kW, stride, stride, pad, pad,
                    cover_all=False, dy=1, dx=1)
    oH, oW = x.shape[-2:]

    x = x.transpose(1, 2, 3, 0, 4, 5)  # (iC, kH, kW, N, oH, oW)
    x = x.reshape(G, iCg * kH * kW, N * oH * oW)

    weights = weights.reshape(G, oCg, iCg * kH * kW)

    # (G, oCg, N*oH*oW) = (G, oCg, iCg*kH*kW) @ (G, iCg*kH*kW, N*oH*oW)
    y = xp.matmul(weights, x)
    y = y.reshape(oC, N, oH, oW)
    y = y.transpose(1, 0, 2, 3)  # (N, oC, oH, oW)
    if bias is not None:
        y += bias.reshape(1, bias.size, 1, 1)

    return y

def relu_copy(x):
    out = x.copy()
    out[x <= 0] = 0
    return out

def relu(x):
    x[x <= 0] = 0
    return x

def batch_normalization(x, gamma, beta, running_mean, running_var):
    N, C, H, W = x.shape
    xc = x - running_mean.reshape(1, C, 1, 1)
    xn = xc / (xp.sqrt(running_var.reshape(1, C, 1, 1) + 1e-5))
    x = gamma.reshape(1, C, 1, 1) * xn + beta.reshape(1, C, 1, 1) 
    return x

def forward(x, weights):
    index = 0
    def arr():
        nonlocal index
        w = weights['arr_' + str(index)]
        index = index + 1
        return w

    # half
    x = convolution2d(x, arr(), stride=2, pad=0)
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    # half
    x = convolution2d(x, arr(), stride=2, pad=0)
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    # half
    x = convolution2d(x, arr(), stride=2, pad=0)
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = depthwise_convolution2d(x, arr(), stride=1, pad=1)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    x = convolution2d(x, arr(), stride=1, pad=0)
    x = batch_normalization(x, arr(), arr(), arr(), arr())
    index = index + 1
    x = relu(x)

    confidence = convolution2d(x, arr(), stride=1, pad=0)
    location = convolution2d(x, arr(), stride=1, pad=0)

    return confidence, location

def load_weights(weights_path):
    weights_file = xp.load(weights_path)
    weights = {}
    for key in weights_file:
        weights[key] = weights_file[key]
    return weights

def softmax(x) :
    max_x = xp.max(x, axis=1, keepdims=True)
    exp_x = xp.exp(x-max_x)
    return exp_x / (xp.sum(exp_x, axis=1, keepdims=True) + 10e-7)

def get_boxes_scores(orig_image, image_size, weights):
    image = orig_image.copy()
    image.thumbnail((image_size, image_size), Image.ANTIALIAS)
    image = xp.array(image)
    image = xp.expand_dims(image, axis=0).transpose(0, 3, 1, 2)
    image = image.astype(xp.float32)
    image = (image - 127) / 128
    scores, locations = forward(image, weights)
    scores = softmax(scores)

    scores = xp.transpose(scores, (0, 2, 3, 1))
    scores = scores.reshape(scores.shape[0], -1, scores.shape[3])

    boxes = convert_locations_to_boxes(
        locations, 0.1, 0.2, image_size
    )
    boxes = xp.transpose(boxes, (0, 2, 3, 1))
    boxes = boxes.reshape(boxes.shape[0], -1, boxes.shape[3])
    boxes = center_form_to_corner_form(boxes)
    boxes = boxes[0]
    scores = scores[0]
    print(scores)
    # scores = scores * 320 / image_size
    return boxes, scores

def predict(orig_image):
    orig_image.thumbnail((640, 640), Image.ANTIALIAS)
    weights = load_weights('weights.npz')
    boxes80,scores80 = get_boxes_scores(orig_image, 80, weights)
    boxes120,scores120 = get_boxes_scores(orig_image, 120, weights)
    boxes140,scores140 = get_boxes_scores(orig_image, 140, weights)
    boxes160,scores160 = get_boxes_scores(orig_image, 160, weights)
    boxes180,scores180 = get_boxes_scores(orig_image, 180, weights)
    boxes200,scores200 = get_boxes_scores(orig_image, 200, weights)
    boxes = xp.concatenate([boxes80, boxes120, boxes140, boxes160, boxes180, boxes200], axis=0)
    scores = xp.concatenate([scores80, scores120, scores140, scores160, scores180, scores200], axis=0)

    # this version of nms is slower on GPU, so we move data to CPU.
    picked_box_probs = []
    picked_labels = []
    probs = scores[:, 0]
    box_probs_all = []
    for class_index in range(1, scores.shape[1]):
        probs = scores[:, class_index]
        mask = xp.where(probs > 0.8)[0]
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = xp.concatenate([subset_boxes, probs.reshape(-1, 1), xp.full((subset_boxes.shape[0],1), class_index)], axis=1)
        print(box_probs)
        box_probs_all.append(box_probs)

    if 0 == len(box_probs_all):
        # print("not found object.\n")
        # sys.exit(0)
        return orig_image

    box_probs = xp.concatenate(box_probs_all, axis=0)
    box_probs = hard_nms(box_probs, 0.20)
    picked_box_probs = box_probs
    print(box_probs.shape)
    picked_labels = box_probs[...,5]
    print(picked_labels)
    width, height = orig_image.size
    max_size = max(width,height)

    print(picked_box_probs.shape)
    picked_box_probs[:, 0] *= max_size
    picked_box_probs[:, 1] *= max_size
    picked_box_probs[:, 2] *= max_size
    picked_box_probs[:, 3] *= max_size
    #if height == width:
    #    pass
    #elif height > width:
    #    picked_box_probs[:, 0] -= (height - width) / 2
    #    picked_box_probs[:, 2] -= (height - width) / 2
    #elif height < width:
    #    picked_box_probs[:, 1] -= (width - height) / 2
    #    picked_box_probs[:, 3] -= (width - height) / 2
    boxes, labels, probs = picked_box_probs[:, :4], picked_labels, picked_box_probs[:, 4]
    print(f"Found {len(probs)} objects.")

    label_path = 'my-model-labels.txt'
    class_names = [name.strip() for name in open(label_path).readlines()]
    draw = ImageDraw.Draw(orig_image)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        print(box)
        if math.isnan(box[0]) or math.isnan(box[1]) or math.isnan(box[2]) or math.isnan(box[3]):
            continue
        if math.isinf(box[0]) or math.isinf(box[1]) or math.isinf(box[2]) or math.isinf(box[3]):
            continue
        if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
            continue
        draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        #label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        label = f"{class_names[int(labels[i])]}"
        draw.text((int(box[0]), int(box[3])), label, fill=(255, 0, 0), font=ImageFont.truetype('GenEiLateGo_v2.ttc', 16))
    return orig_image
