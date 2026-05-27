# main.py
import os
import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import config
from models import ModelWrapper
from managers import PillDecisionManager, ActionDecisionManager

class MedSensingApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x650")
        
        # 初始化模型與管理器
        self.models = ModelWrapper()
        
        # 同步優化：直接綁定 config 唯一的自訂藥單，改 config 全系統同步！
        self.target_meds = config.SELF_DEFINED_MEDS
        self.completed_meds = []
        
        self.pill_manager = PillDecisionManager(self.target_meds, self.completed_meds)
        self.action_manager = ActionDecisionManager()
        
        # 系統狀態控制
        self.mode = "IDLE"  # IDLE, ID_ACTIVE, ACTION_ACTIVE, MATCH_DELAY
        self.camera_id = 1  
        self.vid = None
        self.last_pills = []     # 確保初始化 last_pills 變數
        
        # 💡 新增需求：異常次數計數、超時計時器與異常紀錄本
        self.wrong_count = 0        # 累計拿錯藥的次數（跨輪累積）
        self.wrong_start_time = 0   # 錯誤藥物持續留在畫面的起始時間點
        self.exception_logs = []    # 異常事件儲存陣列
        
        # 💡 新增：單輪內是否已經忽略過錯藥的旗標
        self.wrong_ignored_this_round = False
        
        self.setup_ui()
        self.update_loop()
        
    def setup_ui(self):
        # 左側控制面板
        left_frame = tk.Frame(self.window, width=250, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 狀態提示詞
        self.lbl_tip = tk.Label(left_frame, text="【提示】系統準備就緒", font=("Microsoft JhengHei", 12, "bold"), fg="blue", bg="#f0f0f0", wraplength=220, justify=tk.LEFT)
        self.lbl_tip.pack(anchor=tk.W, pady=10)
        
        # 本次服藥清單區域
        tk.Label(left_frame, text="本日服藥核對清單：", font=("Microsoft JhengHei", 11, "bold"), bg="#f0f0f0").pack(anchor=tk.W, pady=5)
        self.list_frame = tk.Frame(left_frame, bg="#f0f0f0")
        self.list_frame.pack(fill=tk.BOTH, expand=True)
        self.refresh_med_list_ui()
        
        # 按鈕群
        self.btn_start = tk.Button(left_frame, text="開始辨識", font=("Microsoft JhengHei", 11, "bold"), bg="#4CAF50", fg="white", width=18, command=self.start_identification)
        self.btn_start.pack(pady=10)
        
        self.btn_show_photo = tk.Button(left_frame, text="查看藥品正面照", font=("Microsoft JhengHei", 11), width=18, command=self.show_pill_photo_catalog)
        self.btn_show_photo.pack(pady=5)
        
        self.btn_switch_cam = tk.Button(left_frame, text="切換前/後鏡頭", font=("Microsoft JhengHei", 11), width=18, command=self.switch_camera)
        self.btn_switch_cam.pack_forget() 
        
        # 右側視訊畫面
        self.right_frame = tk.Frame(self.window, width=640, height=480, bg="black")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.lbl_video = tk.Label(self.right_frame, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

    def refresh_med_list_ui(self):
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        for med in self.target_meds:
            status = " (已服藥)" if med in self.completed_meds else " (未服藥)"
            color = "green" if med in self.completed_meds else "red"
            lbl = tk.Label(self.list_frame, text=f"• {med}{status}", font=("Microsoft JhengHei", 11), fg=color, bg="#f0f0f0")
            lbl.pack(anchor=tk.W, pady=2)

    def start_identification(self):
        if self.vid is None:
            self.vid = cv2.VideoCapture(self.camera_id)
        self.mode = "ID_ACTIVE"
        self.lbl_tip.config(text="【提示】請拍攝藥物開始偵測", fg="orange")
        self.btn_start.config(state=tk.DISABLED)

    def switch_camera(self):
        self.camera_id = 0 if self.camera_id == 1 else 1
        if self.vid:
            self.vid.release()
        self.vid = cv2.VideoCapture(self.camera_id)

    def show_pill_photo_catalog(self):
        catalog = tk.Toplevel(self.window)
        catalog.title("本次藥單標準藥品正面對照圖鑑")
        catalog.geometry("600x420") 
        catalog.attributes("-topmost", True)  
        
        title_lbl = tk.Label(catalog, text="💡 本次服藥核對照片圖鑑 (等比例對照)", font=("Microsoft JhengHei", 13, "bold"), fg="#333333")
        title_lbl.pack(pady=10)
        
        grid_frame = tk.Frame(catalog)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        catalog.image_refs = []
        GRID_W = 150
        GRID_H = 150

        for idx, (med_name, img_path) in enumerate(config.MED_IMAGE_PATHS.items()):
            row = idx // 3  
            col = idx % 3
            
            item_frame = tk.Frame(grid_frame, bd=1, relief=tk.RIDGE, bg="white", width=GRID_W, height=GRID_H)
            item_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            item_frame.grid_propagate(False) 
            
            content_frame = tk.Frame(item_frame, bg="white")
            content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER) 

            lbl_name = tk.Label(content_frame, text=med_name, font=("Microsoft JhengHei", 9, "bold"), wraplength=GRID_W-20, bg="white", fg="blue")
            lbl_name.pack(pady=(0, 5))

            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path)
                    MAX_SIZE = (120, 100) 
                    pil_img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
                    
                    img_tk = ImageTk.PhotoImage(pil_img)
                    catalog.image_refs.append(img_tk)  
                    
                    lbl_img = tk.Label(content_frame, image=img_tk, bg="white")
                    lbl_img.pack()
                except:
                    lbl_err = tk.Label(content_frame, text="[圖片格式錯誤]", font=("Microsoft JhengHei", 9), fg="red", bg="white")
                    lbl_err.pack(pady=20)
            else:
                lbl_missing = tk.Label(content_frame, text=f"[ 暫無照片 ]\n放入:\n{med_name}.png", 
                                       font=("Microsoft JhengHei", 8), fg="gray", bg="#f0f0f0", width=15, height=5)
                lbl_missing.pack(pady=10)
                
        btn_close = tk.Button(catalog, text="關閉圖鑑", font=("Microsoft JhengHei", 10), bg="#9E9E9E", fg="white", command=catalog.destroy)
        btn_close.pack(side=tk.BOTTOM, pady=10)

    def update_loop(self):
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                h, w, _ = frame.shape
                
                # ----------------------------------------------------
                # 後鏡頭：藥物辨識模式
                # ----------------------------------------------------
                if self.mode == "ID_ACTIVE":
                    results = self.models.pill_yolo(frame, conf=0.45, verbose=False)
                    all_boxes = results[0].boxes if results[0].boxes else []
                    
                    # 💡 修正點 1：如果是多輪累積 >= 2 次，才打從一開始就忽略。
                    # 如果只是單輪內在等 8 秒，一開始先不忽略(傳 False)，這樣 managers 才會回傳 WARNING 讓我們計秒！
                    should_ignore_wrong = self.wrong_ignored_this_round or self.wrong_count >= 3
                    
                    state, res_val = self.pill_manager.process_multi_pills(
                        frame, all_boxes, self.models, ignore_wrong=should_ignore_wrong
                    )
                    
                    # 💡 修正點 2：處理錯誤藥物計時與攔截邏輯
                    if state == "WARNING":
                        box, msg = res_val  
                        
                        # 全新輪次的第一次拿錯，啟動 8 秒計時
                        if self.wrong_start_time == 0:
                            self.wrong_start_time = time.time()
                            self.wrong_count += 1
                            self.exception_logs.append(f"[{time.strftime('%X')}] 異常提示：偵測到錯誤藥物（累積第 {self.wrong_count} 次）。")
                        
                        elapsed = time.time() - self.wrong_start_time
                        
                        if elapsed >= 5.0:
                            # 💡 只有在這裡，真的撐過 8 秒了，才開通「單輪忽略」！
                            self.wrong_ignored_this_round = True
                            self.exception_logs.append(f"[{time.strftime('%X')}] 異常強制：錯誤藥物滯留超時 8 秒，病人未移除，本輪強制忽略該異常並繼續核對。")
                            
                            # 這一幀手動把計時器歸零，下一幀 loop 進來時 should_ignore_wrong 就會是 True，直接解鎖卡死
                            self.wrong_start_time = 0
                        else:
                            # 8 秒之內：老老實實畫紅框、倒數計秒！
                            remaining_sec = int(5.0 - elapsed) + 1
                            self.lbl_tip.config(text=f"【提示】{msg} (剩餘 {remaining_sec} 秒後強制放行)", fg="red")
                            if box:
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3) # 紅框
                    
                    # 如果多輪累積 >=2 次，或者是超時後自動放行，補登錄日誌
                    if should_ignore_wrong and state == "WARNING":
                        if self.wrong_count >= 2 and "多輪累積達 2 次限制" not in "".join(self.exception_logs):
                            self.exception_logs.append(f"[{time.strftime('%X')}] 異常提示：多輪累積拿錯藥已達 {self.wrong_count} 次，系統直接忽略。")
                    
                    # 若畫面中沒有錯誤藥物（例如病人自己拿走了），歸零超時計時器
                    if state != "WARNING":
                        self.wrong_start_time = 0

                    # ----------------------------------------------------
                    # 常規辨識流程（當超時放行或累積達標後，state 就會順利走入這裡）
                    # ----------------------------------------------------
                    if state == "NEED_FLIP":
                        box, _ = res_val
                        self.lbl_tip.config(text="【提示】請將被框取藥品翻面", fg="magenta") 
                        if box: 
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 3) # 橘框
                            
                    elif state == "SCANNING":
                        if should_ignore_wrong:
                            self.lbl_tip.config(text=f"【提示】{res_val} (已忽略錯藥異常)", fg="orange")
                        else:
                            self.lbl_tip.config(text=f"【提示】{res_val}", fg="orange")
                        
                    elif state == "MATCHED":
                        if isinstance(res_val, tuple):
                            self.last_pills, display_text = res_val
                        else:
                            self.last_pills = res_val
                            display_text = f"已識別特徵: {' + '.join(self.last_pills)}"
                        
                        self.lbl_tip.config(text=f"【提示】{display_text}", fg="green")
                        
                        self.mode = "MATCH_DELAY"
                        self.window.after(2000, self.transition_to_action_mode)

                # ----------------------------------------------------
                # 前鏡頭：服藥動作辨識模式
                # ----------------------------------------------------
                elif self.mode == "ACTION_ACTIVE":
                    detected_objs = self.models.process_action(frame)
                    act_stage = self.action_manager.update(detected_objs)
                    
                    if act_stage == 1:
                        self.lbl_tip.config(text="【提示】請向鏡頭展示手掌上有藥品", fg="blue")
                    elif act_stage == 2:
                        self.lbl_tip.config(text="【提示】請捂嘴服藥等待確認", fg="purple")
                    elif act_stage == 3:
                        for p in self.last_pills:
                            if p not in self.completed_meds:
                                self.completed_meds.append(p)
                        self.refresh_med_list_ui()
                        
                        if self.exception_logs:
                            print("\n=== 🧠 本輪服藥異常狀況紀錄公報 ===")
                            for log in self.exception_logs:
                                print(log)
                            print("=====================================\n")
                        
                        self.mode = "MATCH_DELAY"
                        self.window.after(2000, self.auto_reset_to_next_round)

                # 將影像渲染顯示在 Tkinter 畫面上
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_video.imgtk = imgtk
                self.lbl_video.config(image=imgtk)
                
        self.window.after(30, self.update_loop)

    def transition_to_action_mode(self):
        self.camera_id = 0 
        if self.vid: self.vid.release()
        self.vid = cv2.VideoCapture(self.camera_id)
        
        self.mode = "ACTION_ACTIVE"
        self.btn_switch_cam.pack(pady=5) 
        self.action_manager.reset()

    def auto_reset_to_next_round(self):
        self.camera_id = 1 
        if self.vid: self.vid.release()
        self.vid = cv2.VideoCapture(self.camera_id)
        
        self.pill_manager = PillDecisionManager(self.target_meds, self.completed_meds)
        
        # 💡 新一輪重置：單輪忽略旗標重置，超時計時重置，但自我累積的 wrong_count 保留！
        self.wrong_start_time = 0
        self.wrong_ignored_this_round = False
        
        if not self.pill_manager.remaining_targets:
            self.mode = "IDLE"
            self.lbl_tip.config(text="🎉 本日所有藥物皆已核對並服用完畢！", fg="green")
            self.btn_start.config(state=tk.NORMAL) 
            if self.vid: 
                self.vid.release()
                self.vid = None
        else:
            self.mode = "ID_ACTIVE"
            self.lbl_tip.config(text="【提示】請拍攝「剩餘未服」的藥物開始偵測", fg="orange")
            self.btn_switch_cam.pack_forget() 
            
        self.refresh_med_list_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = MedSensingApp(root, "智慧服藥系統")
    root.mainloop()