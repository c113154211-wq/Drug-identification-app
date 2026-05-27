# managers.py
import cv2
import time
import config

def is_overlapping(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) > 0 and max(0, yB - yA) > 0

class PillDecisionManager:
    def __init__(self, target_meds, completed_meds):
        self.remaining_targets = [m for m in target_meds if m not in completed_meds]
        self.completed_meds = completed_meds
        
        # 狀態機核心控制
        self.step_mode = "STEP_1_SCAN"  
        self.wrong_pills_to_warn = []
        
        # 階段一計數器
        self.step1_frames = 0
        self.step1_required_frames = 10
        self.step1_recorded_pills = set()  
        self.saved_flip_box = None
        
        # 翻完面後的 3 秒倒數計時器
        self.post_flip_start_time = 0
        self.flip_delay_duration = 5.0  # 翻完面後等5秒
        
        # 階段二計數器 (第二面)
        self.step2_frames = 0
        self.step2_required_frames = 10

        # 純刻字通道
        self.instant_pass_frames = 0
        self.instant_required_frames = 3  

    # 💡 核心修改：加上 ignore_wrong 參數，預設為 False
    def process_multi_pills(self, frame, all_boxes, model_wrapper, ignore_wrong=False):
        detected_this_frame = []
        self.wrong_pills_to_warn = []
        
        has_smooth_this_frame = False
        current_box = None

        # ----------------------------------------------------
        # 1. 基礎特徵掃描
        # ----------------------------------------------------
        for box_tensor in all_boxes:
            box = box_tensor.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)
            current_box = [x1, y1, x2, y2]
            
            pad = int(max(x2-x1, y2-y1) * 0.15)
            crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad), max(0,x1-pad):min(frame.shape[1],x2+pad)]
            if crop.size == 0: continue
            
            raw_label, conf, diff, crop_pil = model_wrapper.predict_crop(crop)
            if conf < 0.60 or diff < 0.12: continue

            if raw_label in ["Letter", "OneLine"] or raw_label in config.INSTANT_PASS_LIST:
                detected_this_frame.append(raw_label)
            elif raw_label == "Cross":
                color = model_wrapper.get_pill_color(crop_pil)
                lbl = config.CROSS_WHITE_MED if color == "白色" else config.CROSS_GREEN_MED
                detected_this_frame.append(lbl)
            elif raw_label == "Smooth":
                has_smooth_this_frame = True
                if self.step_mode == "STEP_2_SCAN":
                    detected_this_frame.append(config.DOUBLE_SMOOTH_MED)

        # ----------------------------------------------------
        # 2. 異常藥物攔截
        # ----------------------------------------------------
        for label in detected_this_frame:
            if label in config.ALL_19_MEDS and label not in self.remaining_targets and label not in self.completed_meds:
                self.wrong_pills_to_warn.append((current_box, label))
                
        # 💡 核心修改：如果外部傳入 ignore_wrong == True，就不會進到這個 if 攔截器，直接放行去做常規辨識！
        if self.wrong_pills_to_warn and not ignore_wrong:
            # 回傳格式：("WARNING", (第一個錯誤藥物的box, 提示文字))
            first_wrong_box = self.wrong_pills_to_warn[0][0]
            return "WARNING", (first_wrong_box, "請將被框取藥品去除")

        # ----------------------------------------------------
        # 3. 翻面時間差狀態機邏輯
        # ----------------------------------------------------
        
        # 🔷 【階段 1】初次掃描第一面 (看滿 10 幀)
        if self.step_mode == "STEP_1_SCAN":
            if detected_this_frame and not has_smooth_this_frame:
                self.instant_pass_frames += 1
                self.step1_frames = 0 
                if self.instant_pass_frames >= self.instant_required_frames:
                    valid_pills = [m for m in detected_this_frame if m in self.remaining_targets]
                    if valid_pills:
                        self.instant_pass_frames = 0
                        return "MATCHED", valid_pills
                return "SCANNING", f"已辨識刻字藥: {', '.join(detected_this_frame)} (鎖定中...)"

            elif has_smooth_this_frame:
                self.instant_pass_frames = 0 
                self.step1_frames += 1
                
                for label in detected_this_frame:
                    if label in self.remaining_targets:
                        self.step1_recorded_pills.add(label)
                
                # 第一面看滿 10 幀了，立刻鎖定進入 if 分支，提示病人手去翻面
                if self.step1_frames >= self.step1_required_frames:
                    self.saved_flip_box = current_box
                    self.step_mode = "STEP_WAIT_FLIP" # 轉移至等待翻面
                        
            else:
                self.step1_frames = max(0, self.step1_frames - 1)
                self.instant_pass_frames = max(0, self.instant_pass_frames - 1)
                
            pills_str = "+".join(list(self.step1_recorded_pills)) if self.step1_recorded_pills else "等待特徵..."
            return "SCANNING", f"已識別特徵: {pills_str} (第一面鎖定進度: {self.step1_frames}/{self.step1_required_frames})"

        # 🔷 【階段 2】進入 if 分支：提示並等病人動手翻面
        elif self.step_mode == "STEP_WAIT_FLIP":
            if has_smooth_this_frame or detected_this_frame:
                self.post_flip_start_time = time.time() # 記下翻完面的時間點
                self.step_mode = "STEP_POST_FLIP_DELAY" # 切換到 5 秒死等分支
                return "NEED_FLIP", (self.saved_flip_box, "偵測到翻面動作，正在開啟 5 秒穩定緩衝...")
            
            return "NEED_FLIP", (self.saved_flip_box, "第一面鎖定成功！請將藥品進行翻面")

        # 🔷 【階段 3】等病人翻完面之後，在這裡強制死等 5 秒鐘！
        elif self.step_mode == "STEP_POST_FLIP_DELAY":
            elapsed = time.time() - self.post_flip_start_time
            remaining = int(self.flip_delay_duration - elapsed) + 1
            
            if elapsed < self.flip_delay_duration:
                return "SCANNING", f"【翻面完成】請抽離手部，系統將於 {remaining} 秒後開始嚴格審查..."
            else:
                self.step2_frames = 0
                self.step_mode = "STEP_2_SCAN"
                return "SCANNING", "緩衝結束，開始辨識第二面結果..."

        # 🔷 【階段 4】第二面嚴格看完 10 幀
        elif self.step_mode == "STEP_2_SCAN":
            if has_smooth_this_frame:
                self.step2_frames += 1
                
                for label in detected_this_frame:
                    if label in self.remaining_targets:
                        self.step1_recorded_pills.add(label)
                
                if self.step2_frames >= self.step2_required_frames:
                    final_release_pills = list(self.step1_recorded_pills)
                    
                    if config.DOUBLE_SMOOTH_MED in self.remaining_targets:
                        if config.DOUBLE_SMOOTH_MED not in final_release_pills:
                            final_release_pills.append(config.DOUBLE_SMOOTH_MED)
                            
                    if final_release_pills:
                        pills_str = "+".join(final_release_pills)
                        
                        self.step_mode = "STEP_1_SCAN"
                        self.step1_frames = 0
                        self.step2_frames = 0
                        self.instant_pass_frames = 0
                        self.step1_recorded_pills.clear()
                        
                        return "MATCHED", (final_release_pills, f"已識別特徵: {pills_str}！開始服藥動作辨識")
            else:
                self.step2_frames = max(0, self.step2_frames - 1)
                
            return "SCANNING", f"【第二面嚴格核對中】請保持不動 (進度: {self.step2_frames}/{self.step2_required_frames})"

        return "SCANNING", "請拍攝藥物開始偵測"
    
class ActionDecisionManager:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.stage = 1  
        self.stable_frames = 0
        self.required_frames = 5  

    def update(self, detected_objects):
        hands = [obj for obj in detected_objects if obj['label'] == 'hand']
        pills = [obj for obj in detected_objects if obj['label'] == 'pill']
        mouths = [obj for obj in detected_objects if obj['label'] in ['open_mouth', 'closed_mouth']]

        if self.stage == 1:
            has_pill_on_hand = False
            for h in hands:
                for p in pills:
                    if is_overlapping(h['box'], p['box']):
                        has_pill_on_hand = True
                        break
                if has_pill_on_hand: break
                
            if has_pill_on_hand:
                self.stage = 2
                self.stable_frames = 0

        elif self.stage == 2:
            is_covered = False
            if hands and mouths:
                for h in hands:
                    for m in mouths:
                        if is_overlapping(h['box'], m['box']):
                            is_covered = True
                            break
                    if is_covered: break
            
            elif hands and not mouths:
                for h in hands:
                    if h['box'][1] < 240: 
                        is_covered = True
                        break

            if is_covered:
                self.stable_frames += 1
                if self.stable_frames >= self.required_frames:
                    self.stage = 3  
            else:
                self.stable_frames = max(0, self.stable_frames - 1)

        return self.stage