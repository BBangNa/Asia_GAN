import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('./imgs/14.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()

img_result = img.copy()
dets = detector(img) # 얼굴이 여러개있을 수도 있다.
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height() # 얼굴의 왼쪽, 위쪽, 폭과 넓이를 받아서 좌표값을 얻는다.
        rect = patches.Rectangle((x,y),w,h,linewidth=2, edgecolor='r', facecolor='none') # 이를 Rectangle을 통해 사각형으로 만들어준다.
        # linewidth는 사각형 선의 굵기, edgecolor는 선 색깔, facecolor는 사각형을 어떤 색으로 채워주는지에 대한 명령
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show() # --> 지금까지는 얼굴을 찾은 코드

fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()














