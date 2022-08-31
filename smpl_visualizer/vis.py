import numpy as np
import os
import os.path as osp
import subprocess
import platform
import vtk
import cv2 as cv
from PIL import ImageColor
import matplotlib.pyplot as plt


FFMPEG_PATH = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
font_files = {
    'Windows': 'C:/Windows/Fonts/arial.ttf',
    'Linux': '/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
    'Darwin': '/System/Library/Fonts/Supplemental/Arial.ttf'
}


def images_to_video(img_dir, out_path, img_fmt="%06d.png", fps=30, crf=10, verbose=True):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [FFMPEG_PATH, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', '0',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)


def video_to_images(video_path, out_path, img_fmt="%06d.png", fps=30, verbose=True):
    os.makedirs(out_path, exist_ok=True)
    cmd = [FFMPEG_PATH, '-i', video_path, '-r', f'{fps}', f'{out_path}/{img_fmt}']
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)


def hstack_videos(video1_path, video2_path, out_path, crf=10, verbose=True, text1=None, text2=None, text_color='white', text_size=60):
    if not (text1 is None or text2 is None):
        write_text = True
        tmp_file = f'{osp.splitext(out_path)[0]}_tmp.mp4'
    else:
        write_text = False

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [FFMPEG_PATH, '-y', '-i', video1_path, '-i', video2_path, '-filter_complex', 'hstack,format=yuv420p', 
           '-vcodec', 'libx264', '-crf', f'{crf}', tmp_file if write_text else out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)

    if write_text:
        font_file = font_files[platform.system()]
        draw_str = f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text1}':x=(w-text_w)/4:y=20,"\
                   f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text2}':x=3*(w-text_w)/4:y=20" 
        cmd = [FFMPEG_PATH, '-i', tmp_file, '-y', '-vf', draw_str, '-c:a', 'copy', out_path]
        if not verbose:
            cmd += ['-hide_banner', '-loglevel', 'error']
        subprocess.run(cmd)
        os.remove(tmp_file)


def vstack_videos(video1_path, video2_path, out_path, crf=10, verbose=True, text1=None, text2=None, text_color='white', text_size=60):
    if not (text1 is None or text2 is None):
        write_text = True
        tmp_file = f'{osp.splitext(out_path)[0]}_tmp.mp4'
    else:
        write_text = False

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [FFMPEG_PATH, '-y', '-i', video1_path, '-i', video2_path, '-filter_complex', 'vstack,format=yuv420p', 
           '-vcodec', 'libx264', '-crf', f'{crf}', tmp_file if write_text else out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)

    if write_text:
        font_file = font_files[platform.system()]
        draw_str = f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text1}':x=10:y=20,"\
                   f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text2}':x=10:y=h/2+20" 
        cmd = [FFMPEG_PATH, '-i', tmp_file, '-y', '-vf', draw_str, '-c:a', 'copy', out_path]
        if not verbose:
            cmd += ['-hide_banner', '-loglevel', 'error']
        subprocess.run(cmd)
        os.remove(tmp_file)



def make_checker_board_texture(color1='black', color2='white', width=1000, alpha=None):
    c1 = np.asarray(ImageColor.getcolor(color1, 'RGB')).astype(np.uint8)
    c2 = np.asarray(ImageColor.getcolor(color2, 'RGB')).astype(np.uint8)
    if alpha is not None:
        c1 = np.append(c1, int(alpha*255))
        c2 = np.append(c2, int(alpha*255))
    hw = hh = width // 2
    c1_block = np.tile(c1, (hh, hw, 1))
    c2_block = np.tile(c2, (hh, hw, 1))
    tex = np.block([
        [[c1_block], [c2_block]],
        [[c2_block], [c1_block]]
    ])
    return tex
    

def resize_bbox(bbox, scale):
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    h, w = y2 - y1, x2 - x1
    cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
    h_new, w_new = h * scale, w * scale
    x1_new, x2_new = cx - 0.5 * w_new, cx + 0.5 * w_new
    y1_new, y2_new = cy - 0.5 * h_new, cy + 0.5 * h_new
    bbox_new = np.stack([x1_new, y1_new, x2_new, y2_new], axis=-1)
    return bbox_new


def nparray_to_vtk_matrix(array):
    """Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    matrix = vtk.vtkMatrix4x4()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            matrix.SetElement(i, j, array[i, j])
    return matrix


def vtk_matrix_to_nparray(matrix):
    """Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    array = np.zeros([4, 4])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = matrix.GetElement(i, j)
    return array


def draw_keypoints(img, keypoints, confidence, size=4, color=(255, 0, 255)):
    for kp, conf in zip(keypoints, confidence):
        if conf > 0.2:
            cv.circle(img, np.round(kp).astype(int).tolist(), size, color=color, thickness=-1)
    return img


def get_color_palette(n, colormap='rainbow', use_float=False):
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    unit = 1 if use_float else 255
    colors = [[int(c[0] * unit), int(c[1] * unit), int(c[2] * unit)] for c in colors]
    return colors