# models.py
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from torchvision import models as tv_models, transforms
from ultralytics import YOLO
import config

class ModelWrapper:
    def __init__(self):
        self.pill_yolo = YOLO(config.YOLO_PILL_PATH)
        self.act_yolo = YOLO(config.YOLO_ACT_PATH)
        self.act_classes = self.act_yolo.names
        
        # 載入 ResNet
        self.resnet = tv_models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, len(config.CLASS_NAMES))
        self.resnet.load_state_dict(torch.load(config.RESNET_PATH, map_location=config.DEVICE))
        self.resnet.to(config.DEVICE).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_crop(self, crop):
        """
        💡 核心封裝：輸入單顆藥丸的 BGR 影像，自動進行影像增強與 ResNet 預測。
        回傳：標籤名稱 (label), 最高信心度 (conf), 第一與第二名差值 (diff), 處理後的 PIL 影像
        """
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_pil = self.pad_to_square(self.enhance_embossment(pil_img))
        
        with torch.no_grad():
            output = self.resnet(self.transform(crop_pil).unsqueeze(0).to(config.DEVICE))
            prob = torch.nn.functional.softmax(output[0], dim=0)
            top_probs, top_idxs = torch.topk(prob, 2)
            
            label = config.CLASS_NAMES[top_idxs[0].item()]
            conf = top_probs[0].item()
            diff = conf - top_probs[1].item()
            
        return label, conf, diff, crop_pil

    def process_action(self, frame):
        results = self.act_yolo(frame, conf=0.45, verbose=False)
        detected_objects = []
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                label = self.act_classes[cls_id] 
                xyxy = box.xyxy.cpu().numpy()[0] 
                
                detected_objects.append({
                    'label': label,
                    'box': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                })
        return detected_objects

    @staticmethod
    def pad_to_square(pil_img):
        w, h = pil_img.size
        max_dim = max(w, h)
        new_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        new_img.paste(pil_img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return new_img

    @staticmethod
    def enhance_embossment(pil_img):
        return ImageEnhance.Contrast(pil_img).enhance(1.5)

    @staticmethod
    def get_pill_color(crop_pil):
        img_hsv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2HSV)
        h, w, _ = img_hsv.shape
        cy, cx = h // 2, w // 2
        dy, dx = int(h * 0.35), int(w * 0.35)
        core_zone = img_hsv[cy-dy:cy+dy, cx-dx:cx+dx]
        
        avg_s = np.mean(core_zone[:, :, 1])
        return "白色" if avg_s < 45 else "綠色" 

    @staticmethod
    def draw_chinese_text(img, text, position, color=(255, 255, 255), size=28):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "C:/Windows/Fonts/msjh.ttc" if os.name == 'nt' else "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        try: font = ImageFont.truetype(font_path, size)
        except: font = ImageFont.load_default()
        
        draw.text(position, text, font=font, fill=color, spacing=6)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)