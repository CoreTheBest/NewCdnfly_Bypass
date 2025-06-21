import os
import time
import cv2
import numpy as np
import torch
from DrissionPage import ChromiumPage, ChromiumOptions
from model import SiameseNet

# --- 初始化浏览器
co = ChromiumOptions()
co.set_argument('--no-sandbox')
co.set_argument('--disable-gpu')
page = ChromiumPage(co)

# --- 配置
resize_size = (96, 96)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 加载模型
yolo_model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
siamese_model = SiameseNet().to(device)
siamese_model.load_state_dict(torch.load('siamese_model.pt', map_location=device))
siamese_model.eval()

# === 工具函数 ===
def ensure_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

def crop_right_blank_strict(img, thresh=245):
    gray = ensure_gray(img)
    h, w = gray.shape
    for x in range(w - 1, -1, -1):
        if np.min(gray[:, x]) < thresh:
            return img[:, :x + 1]
    return img

def split_small_image_fixed_fuzzy_margin(img, expand_margin=10):
    h, w, _ = img.shape
    step = w // 3
    results = []
    for i in range(3):
        x_start = i * step
        x_end = (i + 1) * step if i < 2 else w
        sub_img = img[:, x_start:x_end]
        gray = ensure_gray(sub_img)
        _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x_min = max(0, min(c[0] for c in [cv2.boundingRect(c) for c in contours]) - expand_margin)
            y_min = max(0, min(c[1] for c in [cv2.boundingRect(c) for c in contours]) - expand_margin)
            x_max = min(sub_img.shape[1], max(c[0] + c[2] for c in [cv2.boundingRect(c) for c in contours]) + expand_margin)
            y_max = min(sub_img.shape[0], max(c[1] + c[3] for c in [cv2.boundingRect(c) for c in contours]) + expand_margin)
            cropped = sub_img[y_min:y_max, x_min:x_max]
        else:
            cropped = sub_img.copy()
        results.append((f"D{i+1}", cropped))
    return results

def split_word_image_to_parts(img, output_dir):
    img = crop_right_blank_strict(img)
    splits = split_small_image_fixed_fuzzy_margin(img, expand_margin=10)
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for idx, (label, crop) in enumerate(splits):
        gray = ensure_gray(crop)
        path = os.path.join(output_dir, f"{idx}_{label}.png")
        cv2.imwrite(path, gray)
        resized = cv2.resize(gray, (64, 64))
        results.append((label, resized))
    return results

def color_to_black_white(img, target_color, tolerance=30):
    lower = np.clip(target_color - tolerance, 0, 255)
    upper = np.clip(target_color + tolerance, 0, 255)
    mask = cv2.inRange(img, lower, upper)
    result = np.ones_like(img) * 255
    result[mask > 0] = (0, 0, 0)
    return result

def preprocess_img_tensor(img):
    img = cv2.resize(img, resize_size)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

# --- 主程序 ---
save_dir = 'captcha_results'
os.makedirs(save_dir, exist_ok=True)

try:
    page.get('/click')
    time.sleep(2)

    # 截图 + 小图切割
    word_img = cv2.imdecode(np.frombuffer(page.ele('css:.word').get_screenshot(as_bytes='png'), np.uint8), cv2.IMREAD_COLOR)
    word_img = cv2.resize(word_img, (word_img.shape[1]*2, word_img.shape[0]*2))
    cv2.imwrite(os.path.join(save_dir, 'word_image_raw.png'), word_img)
    small_parts_dir = os.path.join(save_dir, 'small_parts')
    small_crops = split_word_image_to_parts(word_img, small_parts_dir)

    # 验证码图 + 黑白处理
    captcha_img = cv2.imdecode(np.frombuffer(page.ele('css:.clickableImg').get_screenshot(as_bytes='png'), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(save_dir, 'captcha_image_raw.png'), captcha_img)
    filtered = color_to_black_white(captcha_img, np.array([124, 160, 31]), tolerance=30)
    cv2.imwrite(os.path.join(save_dir, 'captcha_bw_stage1.png'), filtered)

    # YOLO 检测字块
    results = yolo_model(filtered)
    detections = []
    for idx, box in enumerate(results.xyxy[0]):
        if box[4] < 0.3:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        detections.append({'id': idx, 'coords': (x1, y1, x2, y2), 'center': center})

    print(f"✅ 检测到 {len(detections)} 个字块")
    if len(detections) < 3:
        raise Exception("❌ 检测字块太少")

    # 裁剪 block
    block_dir = os.path.join(save_dir, 'blocks')
    os.makedirs(block_dir, exist_ok=True)
    processed_crops = []
    for det in detections:
        x1, y1, x2, y2 = det['coords']
        crop = filtered[y1:y2, x1:x2]
        processed_crops.append({'id': det['id'], 'center': det['center'], 'coords': det['coords'], 'crop': crop})
        cv2.imwrite(os.path.join(block_dir, f"block_{det['id']}.png"), crop)

    # Siamese 匹配，保留得分最高的前三个 block
# Siamese 匹配，保留得分最高的前三个 block
    template_dict = {label: preprocess_img_tensor(crop) for label, crop in small_crops}
    click_sequence = []
    used_ids = set()  # 新增：记录已使用的 block id

    for label in ['D1', 'D2', 'D3']:
        template_tensor = template_dict[label]
        best_score = -1
        best_det = None
        scores_log = []

        for det in processed_crops:
            if det['id'] in used_ids:  # 跳过已使用的位置
                continue

            crop_tensor = preprocess_img_tensor(det['crop'])
            with torch.no_grad():
                score = siamese_model(template_tensor, crop_tensor).item()
            scores_log.append(f"b{det['id']}-{score:.4f}")
            if score > best_score:
                best_score = score
                best_det = det

        print(f"{label} 匹配结果: {' | '.join(scores_log)}")

        if best_det:
            used_ids.add(best_det['id'])  # 新增：标记此位置已用
            click_sequence.append({'label': label, 'center': best_det['center']})


    # --- 绘图 ---
    draw_img = captcha_img.copy()
    for idx, item in enumerate(click_sequence, 1):
        cx, cy = item['center']
        label = item['label']
        cv2.circle(draw_img, (cx, cy), 20, (0, 0, 255), 3)
        cv2.putText(draw_img, label, (cx-25, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    result_path = os.path.join(save_dir, 'click_points_result.png')
    cv2.imwrite(result_path, draw_img)
    print(f"✅ 点击点绘制完成，保存为：{result_path}")


except Exception as e:
    print(f"❌ 出错: {e}")
