import time
import cv2
import numpy as np
from DrissionPage import Chromium
import torch

#初始化
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
names_map = model.names

chrom = Chromium()
tab = chrom.new_tab()
tab.get('测试的url')
time.sleep(1.2)  

def fetch_and_detect(tab, model, names_map):
    img_ele = tab.ele('.captcha-image')
    if not img_ele:
        return None, '[错误] 未找到验证码图片元素'
    img_bytes = img_ele.get_screenshot(as_bytes='png')
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(img_np)
    # 后处理...
    char_map = {
        'n_0': '0', 'n_1': '1', 'n_2': '2', 'n_3': '3', 'n_4': '4',
        'n_5': '5', 'n_6': '6', 'n_7': '7', 'n_8': '8', 'n_9': '9',
        'p_-': '-', 'p_X': '*', 'p_add': '+', 'p_dengyu': '=',
        'p_idk': '?', 'slider': '', 'text': '',
    }
    boxes = results.xyxy[0].cpu().numpy()
    confs = results.xyxy[0][:, 4].cpu().numpy()
    clses = results.xyxy[0][:, 5].cpu().numpy()
    mask = confs > 0.2
    boxes, clses = boxes[mask], clses[mask]
    detected = []
    for box, c in zip(boxes, clses):
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2
        label = names_map[int(c)]
        char = char_map.get(label, '')
        if char:
            detected.append((cx, char))
    detected.sort(key=lambda x: x[0])
    expr = ''.join([char for _, char in detected])
    expr4eval = expr.replace('=', '').replace('?', '')
    try:
        result = eval(expr4eval)
    except Exception as e:
        result = None
    return expr, result

# 用法示例
expr, result = fetch_and_detect(tab, model, names_map)
print('识别:', expr, '| 计算:', result)

tab.close()
chrom.quit()
