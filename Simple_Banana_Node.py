import json
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from typing import List, Dict, Optional
import time
import tempfile
import os
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 独立提取版：香蕉绘画节点 (无授权验证 + Key管理 + 价格表)
# ==============================================================================

class SimpleBananaGenNode:
    
    def __init__(self):
        self.MODEL_MAP = {
            "香蕉1": "nano-banana-fast",
            "香蕉2": "nano-banana-pro", 
            "香蕉2独立渠道": "nano-banana-pro-vt"
        }
        # 获取当前文件所在目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.current_dir, "config.ini")

    @classmethod
    def INPUT_TYPES(cls):
        model_display_names = ["香蕉1", "香蕉2", "香蕉2独立渠道"]
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的猫咪"
                }),
                "model_select": (model_display_names, {"default": "香蕉1"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", 
                                "3:2", "2:3", "5:4", "4:5", "21:9"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "api_key_input": ("STRING", {
                    "default": "", 
                    "multiline": False, 
                    "placeholder": "在此输入API Key (自动保存到本地)"
                }),
                "seed": ("INT", {"default": -1}),
                "image_size": (["1K", "2K", "4K"], {"default": "1K"}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 8}),
                "input_image_1": ("IMAGE",), 
                "input_image_2": ("IMAGE",),
                "webhook_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "generate_images"
    OUTPUT_NODE = True
    CATEGORY = "Banana/Simple"

    # --- Key 管理逻辑 ---
    def _manage_api_key(self, input_key):
        config = configparser.ConfigParser()
        try:
            config.read(self.config_path, encoding='utf-8-sig')
        except:
            config.read(self.config_path, encoding='utf-8')

        final_key = ""

        # 1. 如果用户在 UI 输入了 Key，保存并使用
        if input_key and input_key.strip():
            final_key = input_key.strip()
            if not config.has_section('AUTH'):
                config.add_section('AUTH')
            config.set('AUTH', 'api_key', final_key)
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    config.write(f)
                print(f"💾 [Banana] API Key 已更新并保存到 config.ini")
            except Exception as e:
                print(f"⚠️ [Banana] 保存 config.ini 失败: {e}")
        
        # 2. 如果 UI 为空，尝试从文件读取
        else:
            if config.has_section('AUTH') and config.has_option('AUTH', 'api_key'):
                final_key = config.get('AUTH', 'api_key').strip()
            elif config.has_section('grsai') and config.has_option('grsai', 'api_key'):
                final_key = config.get('grsai', 'api_key').strip()
        
        return final_key

    # --- 警告图片生成 (包含最新价格表) ---
    def _create_warning_image(self):
        w, h = 800, 800
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 字体加载逻辑
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
        try:
            font_names = ["msyh.ttc", "simhei.ttf", "arialuni.ttf", "PingFang.ttc", "STHeiti Medium.ttc"]
            font_path = None
            import platform
            sys_fonts = []
            if platform.system() == "Windows":
                sys_fonts = [os.path.join("C:\\Windows\\Fonts", f) for f in font_names]
            
            for fp in sys_fonts:
                if os.path.exists(fp):
                    font_path = fp
                    break
            
            if not font_path:
                 local_font = os.path.join(os.path.dirname(__file__), "simhei.ttf")
                 if os.path.exists(local_font):
                     font_path = local_font

            if font_path:
                font_title = ImageFont.truetype(font_path, 45) # 标题大字
                font_text = ImageFont.truetype(font_path, 30)  # 正文
                font_small = ImageFont.truetype(font_path, 26) # 价格小字
            else:
                font_title = ImageFont.truetype("simhei.ttf", 45)
                font_text = ImageFont.truetype("simhei.ttf", 30)
                font_small = ImageFont.truetype("simhei.ttf", 26)
        except:
            pass

        # === 内容排版 ===
        content_layout = [
            ("⚠️ 未检测到 API Key", (255, 0, 0), font_title, 10),
            ("请在节点输入框填写 Key", (50, 50, 50), font_text, 5),
            ("如果没有 Key，请联系：", (50, 50, 50), font_text, 5),
            ("微信：C7777666", (220, 20, 60), font_title, 20),
            ("---------------------------------------", (200, 200, 200), font_text, 10),
            ("【 价格公示 】", (0, 0, 0), font_text, 10),
            ("🍌 蕉1价格：0.8分钱/张", (34, 139, 34), font_small, 5),
            ("🍌 蕉2价格：0.2元/张", (34, 139, 34), font_small, 5),
            ("🎬 sora-2价格：0.2元/次", (34, 139, 34), font_small, 5),
            ("🎥 veo3.1-pro价格：4元/次", (34, 139, 34), font_small, 5),
            ("⚡ veo3.1-fast价格：0.9元/次", (34, 139, 34), font_small, 5),
        ]

        total_content_height = 0
        for text, _, font, gap in content_layout:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                h_line = bbox[3] - bbox[1]
                total_content_height += h_line + gap
            except:
                total_content_height += 40

        current_y = (h - total_content_height) / 2 
        if current_y < 50: current_y = 50

        for text, color, font, gap in content_layout:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                x = (w - text_w) / 2
                draw.text((x, current_y), text, font=font, fill=color)
                current_y += text_h + gap
            except:
                pass 
        
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    # --- 辅助函数 ---
    def _get_headers(self, api_key: str) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'ComfyUI-Banana-Simple/1.0'
        }

    def url_to_tensor_single(self, url: str) -> np.ndarray:
        try:
            response = requests.get(url, timeout=15)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return np.array(img).astype(np.float32) / 255.0
        except:
            return np.zeros((64, 64, 3), dtype=np.float32)

    def urls_to_tensor_parallel(self, urls: List[str]) -> torch.Tensor:
        if not urls: return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        with ThreadPoolExecutor(max_workers=min(4, len(urls))) as executor:
            results = list(executor.map(self.url_to_tensor_single, urls))
        return torch.from_numpy(np.stack(results))

    def _upload_image(self, api_key, image_tensor, api_base_url):
        try:
            if image_tensor is None: return None
            if len(image_tensor.shape) == 4: img_array = image_tensor[0].cpu().numpy()
            else: img_array = image_tensor.cpu().numpy()
            img_array = np.clip(255. * img_array, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array).convert('RGB')
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
                pil_img.save(temp_f, "PNG", quality=95)
                temp_f_path = temp_f.name

            headers = self._get_headers(api_key)
            token_url = f"{api_base_url.rstrip('/')}/client/resource/newUploadTokenZH"
            res = requests.post(token_url, headers=headers, json={"sux": "png"}, timeout=30)
            data = res.json().get("data")
            
            with open(temp_f_path, "rb") as f:
                requests.post(data["url"], data={"token": data["token"], "key": data["key"]}, files={"file": f}, timeout=120)
            
            os.unlink(temp_f_path)
            return f"{data['domain']}/{data['key']}"
        except Exception as e:
            print(f"Upload failed: {e}")
            return None

    def process_reference_images(self, api_key, api_base_url, input_images):
        valid = [img for img in input_images if img is not None]
        if not valid: return []
        urls = []
        for img in valid:
            u = self._upload_image(api_key, img, api_base_url)
            if u: urls.append(u)
        return urls

    def create_request_data(self, prompt, seed, aspect_ratio, api_model_name, image_size, reference_urls, webhook_url):
        req = {
            "model": api_model_name,
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "shutProgress": False,
            "urls": reference_urls
        }
        if "pro" in api_model_name:
            req["imageSize"] = image_size
        if seed != -1:
            req["seed"] = seed
        if webhook_url:
            req["webHook"] = webhook_url
        return req

    def send_request(self, api_key, request_data, api_base_url):
        url = f"{api_base_url.rstrip('/')}/v1/draw/nano-banana"
        headers = self._get_headers(api_key)
        
        req = request_data.copy()
        req["webHook"] = "-1"
        
        res = requests.post(url, headers=headers, json=req, timeout=120)
        if res.status_code != 200: raise Exception(f"HTTP Error: {res.status_code}")
        
        res_json = res.json()
        if res_json.get("code") != 0: raise Exception(f"API Error: {res_json.get('msg')}")
        
        task_id = res_json["data"]["id"]
        start = time.time()
        while time.time() - start < 400:
            poll = requests.post(f"{api_base_url.rstrip('/')}/v1/draw/result", headers=headers, json={"id": task_id}, timeout=30).json()
            status = poll.get("data", {}).get("status")
            if status == "succeeded": return poll.get("data")
            elif status == "failed": raise Exception(poll.get("data", {}).get("failure_reason"))
            time.sleep(3)
        raise Exception("Timeout")

    def extract_content(self, data):
        urls = [r['url'] for r in data.get('results', []) if r.get('url')]
        return urls, ""

    def generate_single_image(self, args):
        i, seed, key, prompt, api_model_name, ratio, size, refs, url, hook = args
        try:
            req = self.create_request_data(prompt, seed, ratio, api_model_name, size, refs, hook)
            res = self.send_request(key, req, url)
            urls, txt = self.extract_content(res)
            return {'success': True, 'images': urls, 'text': txt}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # --- 主入口 ---
    def generate_images(self, prompt, model_select, batch_size, aspect_ratio, api_key_input="", seed=-1, image_size="1K", max_workers=4, **kwargs):
        api_model_name = self.MODEL_MAP.get(model_select, "nano-banana-fast")
        api_base_url = "https://grsai.dakka.com.cn"

        # 1. 移除了 check_auth_status 调用，不再检查机器授权
        
        # 2. Key 管理
        api_key = self._manage_api_key(api_key_input)

        # 3. 如果没有 Key，返回警告图
        if not api_key:
            warning_img = self._create_warning_image()
            return (warning_img, "❌ 错误：缺少 API Key，请在节点输入或购买。")

        inputs = [kwargs.get(f"input_image_{i}") for i in range(1, 3)]
        refs = self.process_reference_images(api_key, api_base_url, inputs)

        all_urls = []
        all_txt = []

        if batch_size > 1:
            tasks = []
            for i in range(batch_size):
                s = seed + i if seed != -1 else -1
                tasks.append((i, s, api_key, prompt, api_model_name, aspect_ratio, image_size, refs, api_base_url, kwargs.get("webhook_url")))
            
            with ThreadPoolExecutor(max_workers=min(max_workers, batch_size)) as ex:
                futures = [ex.submit(self.generate_single_image, t) for t in tasks]
                for f in as_completed(futures):
                    r = f.result()
                    if r['success']: all_urls.extend(r['images'])
                    else: all_txt.append(f"Error: {r['error']}")
        else:
            try:
                req = self.create_request_data(prompt, seed, aspect_ratio, api_model_name, image_size, refs, kwargs.get("webhook_url"))
                res = self.send_request(api_key, req, api_base_url)
                u, t = self.extract_content(res)
                all_urls.extend(u)
            except Exception as e:
                all_txt.append(str(e))

        if not all_urls:
            return (torch.zeros((1,64,64,3)), "\n".join(all_txt))

        return (self.urls_to_tensor_parallel(all_urls), "\n".join(all_txt))