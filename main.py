import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import asyncio
import websockets
import json
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List
from queue import Queue
import os
import urllib.request

# ============================================================
# BlendShape 데이터 클래스
# ============================================================
@dataclass
class BlendShapeData:
    # 고개 회전
    head_x: float = 0.0
    head_y: float = 0.0
    head_z: float = 0.0
    
    # 눈
    eye_left_open: float = 1.0
    eye_right_open: float = 1.0
    eye_left_x: float = 0.0
    eye_left_y: float = 0.0
    eye_right_x: float = 0.0
    eye_right_y: float = 0.0
    
    # 입
    mouth_open: float = 0.0
    mouth_smile: float = 0.0
    
    # 눈썹
    brow_left_y: float = 0.0
    brow_right_y: float = 0.0
    
    # 추가 표정
    cheek_puff: float = 0.0
    tongue_out: float = 0.0


# ============================================================
# 랜드마크 인덱스
# ============================================================
class LandmarkIndex:
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263


# ============================================================
# BlendShape 계산기 (face_landmarker.task 사용)
# ============================================================
class BlendShapeCalculator:
    def __init__(self, model_path: str = "face_landmarker.task"):
        self.model_path = model_path
        self._ensure_model()
        
        # FaceLandmarker 옵션 설정
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # 스무딩
        self.prev_data = BlendShapeData()
        self.smoothing_factor = 0.4  # 0.0 ~ 1.0 (높을수록 빠른 반응)
        
        print("✓ FaceLandmarker 초기화 완료")
    
    def _ensure_model(self):
        """모델 파일이 없으면 다운로드"""
        if not os.path.exists(self.model_path):
            print("📥 모델 다운로드 중...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("✓ 모델 다운로드 완료")
    
    def _get_blendshape(self, blendshapes: list, name: str) -> float:
        """BlendShape 값 가져오기"""
        for bs in blendshapes:
            if bs.category_name == name:
                return bs.score
        return 0.0
    
    def _calculate_head_rotation(self, landmarks, w: int, h: int) -> tuple:
        """고개 회전 계산"""
        def get_point(idx):
            lm = landmarks[idx]
            return (lm.x * w, lm.y * h, lm.z * w)
        
        nose = get_point(LandmarkIndex.NOSE_TIP)
        left_cheek = get_point(LandmarkIndex.LEFT_CHEEK)
        right_cheek = get_point(LandmarkIndex.RIGHT_CHEEK)
        forehead = get_point(LandmarkIndex.FOREHEAD)
        chin = get_point(LandmarkIndex.CHIN)
        left_eye = get_point(LandmarkIndex.LEFT_EYE_OUTER)
        right_eye = get_point(LandmarkIndex.RIGHT_EYE_OUTER)
        
        # Yaw (좌우)
        total_width = right_cheek[0] - left_cheek[0]
        if total_width > 0:
            left_dist = nose[0] - left_cheek[0]
            right_dist = right_cheek[0] - nose[0]
            yaw = ((left_dist - right_dist) / total_width) * 30
        else:
            yaw = 0
        
        # Pitch (상하)
        face_height = chin[1] - forehead[1]
        if face_height > 0:
            nose_ratio = (nose[1] - forehead[1]) / face_height
            pitch = (nose_ratio - 0.35) * 50
        else:
            pitch = 0
        
        # Roll (기울임)
        eye_diff = left_eye[1] - right_eye[1]
        eye_width = right_eye[0] - left_eye[0]
        if eye_width > 0:
            roll = np.degrees(np.arctan2(eye_diff, eye_width))
        else:
            roll = 0
        
        return (
            np.clip(yaw, -30, 30),
            np.clip(pitch, -30, 30),
            np.clip(roll, -30, 30)
        )
    
    def _smooth(self, current: float, previous: float) -> float:
        """값 스무딩"""
        return previous + (current - previous) * self.smoothing_factor
    
    def process_frame(self, frame: np.ndarray) -> Optional[BlendShapeData]:
        """프레임 처리 및 BlendShape 추출"""
        h, w = frame.shape[:2]
        
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image 생성
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        
        # 얼굴 감지
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        landmarks = result.face_landmarks[0]
        
        # 고개 회전 계산
        head_x, head_y, head_z = self._calculate_head_rotation(landmarks, w, h)
        
        # BlendShape 추출
        if result.face_blendshapes:
            bs = result.face_blendshapes[0]
            
            # 눈 깜빡임 (1에서 빼서 열림 정도로 변환)
            eye_left_open = 1.0 - self._get_blendshape(bs, "eyeBlinkLeft")
            eye_right_open = 1.0 - self._get_blendshape(bs, "eyeBlinkRight")
            
            # 시선 방향
            eye_left_x = (self._get_blendshape(bs, "eyeLookOutLeft") - 
                         self._get_blendshape(bs, "eyeLookInLeft"))
            eye_left_y = (self._get_blendshape(bs, "eyeLookDownLeft") - 
                         self._get_blendshape(bs, "eyeLookUpLeft"))
            eye_right_x = (self._get_blendshape(bs, "eyeLookInRight") - 
                          self._get_blendshape(bs, "eyeLookOutRight"))
            eye_right_y = (self._get_blendshape(bs, "eyeLookDownRight") - 
                          self._get_blendshape(bs, "eyeLookUpRight"))
            
            # 입
            mouth_open = self._get_blendshape(bs, "jawOpen")
            mouth_smile = (self._get_blendshape(bs, "mouthSmileLeft") + 
                          self._get_blendshape(bs, "mouthSmileRight")) / 2
            
            # 눈썹
            brow_left_y = (self._get_blendshape(bs, "browOuterUpLeft") - 
                          self._get_blendshape(bs, "browDownLeft"))
            brow_right_y = (self._get_blendshape(bs, "browOuterUpRight") - 
                           self._get_blendshape(bs, "browDownRight"))
            
            # 추가 표정
            cheek_puff = (self._get_blendshape(bs, "cheekPuff"))
            tongue_out = self._get_blendshape(bs, "tongueOut")
        else:
            eye_left_open = eye_right_open = 1.0
            eye_left_x = eye_left_y = eye_right_x = eye_right_y = 0.0
            mouth_open = mouth_smile = 0.0
            brow_left_y = brow_right_y = 0.0
            cheek_puff = tongue_out = 0.0
        
        # 스무딩 적용
        data = BlendShapeData(
            head_x=self._smooth(head_x, self.prev_data.head_x),
            head_y=self._smooth(head_y, self.prev_data.head_y),
            head_z=self._smooth(head_z, self.prev_data.head_z),
            eye_left_open=self._smooth(eye_left_open, self.prev_data.eye_left_open),
            eye_right_open=self._smooth(eye_right_open, self.prev_data.eye_right_open),
            eye_left_x=self._smooth(eye_left_x, self.prev_data.eye_left_x),
            eye_left_y=self._smooth(eye_left_y, self.prev_data.eye_left_y),
            eye_right_x=self._smooth(eye_right_x, self.prev_data.eye_right_x),
            eye_right_y=self._smooth(eye_right_y, self.prev_data.eye_right_y),
            mouth_open=self._smooth(mouth_open, self.prev_data.mouth_open),
            mouth_smile=self._smooth(mouth_smile, self.prev_data.mouth_smile),
            brow_left_y=self._smooth(brow_left_y, self.prev_data.brow_left_y),
            brow_right_y=self._smooth(brow_right_y, self.prev_data.brow_right_y),
            cheek_puff=self._smooth(cheek_puff, self.prev_data.cheek_puff),
            tongue_out=self._smooth(tongue_out, self.prev_data.tongue_out),
        )
        
        self.prev_data = data
        return data
    
    def get_raw_blendshapes(self, frame: np.ndarray) -> Dict[str, float]:
        """모든 BlendShape 원본 값 반환 (디버그용)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        
        if result.face_blendshapes:
            return {bs.category_name: bs.score for bs in result.face_blendshapes[0]}
        return {}
    
    def close(self):
        """리소스 해제"""
        self.landmarker.close()


# ============================================================
# VTube Studio API
# ============================================================
class VTubeStudioAPI:
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.authenticated = False
        self.plugin_name = "MediaPipe Tracker"
        self.plugin_developer = "Python"
        self.auth_token = None
        self.request_id = 0
    
    async def connect(self) -> bool:
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"✓ VTube Studio 연결: {self.uri}")
            return True
        except Exception as e:
            print(f"✗ 연결 실패: {e}")
            return False
    
    async def _send(self, msg_type: str, data: dict = None) -> dict:
        self.request_id += 1
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"req_{self.request_id}",
            "messageType": msg_type,
            "data": data or {}
        }
        await self.websocket.send(json.dumps(request))
        return json.loads(await self.websocket.recv())
    
    async def authenticate(self, saved_token: str = None) -> bool:
        # 토큰 요청
        if not saved_token:
            resp = await self._send("AuthenticationTokenRequest", {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "pluginIcon": ""
            })
            self.auth_token = resp.get("data", {}).get("authenticationToken")
            if self.auth_token:
                print("⏳ VTube Studio에서 플러그인을 승인해주세요...")
                await asyncio.sleep(3)
        else:
            self.auth_token = saved_token
        
        # 인증
        resp = await self._send("AuthenticationRequest", {
            "pluginName": self.plugin_name,
            "pluginDeveloper": self.plugin_developer,
            "authenticationToken": self.auth_token
        })
        
        self.authenticated = resp.get("data", {}).get("authenticated", False)
        print(f"{'✓' if self.authenticated else '✗'} 인증 {'성공' if self.authenticated else '실패'}")
        return self.authenticated
    
    async def send_tracking_data(self, data: BlendShapeData):
        if not self.authenticated:
            return

        # 눈썹 전체 (좌우 평균)
        brows = (data.brow_left_y + data.brow_right_y) * 0.5

        params = [
            # ======================
            # 얼굴 각도
            # ======================
            {"id": "FaceAngleX", "value": data.head_x},  # Pitch
            {"id": "FaceAngleY", "value": data.head_y},  # Yaw
            {"id": "FaceAngleZ", "value": data.head_z},  # Roll

            # ======================
            # 눈
            # ======================
            {"id": "EyeOpenLeft",  "value": data.eye_left_open},
            {"id": "EyeOpenRight", "value": data.eye_right_open},

            {"id": "EyeLeftX",  "value": data.eye_left_x},
            {"id": "EyeLeftY",  "value": data.eye_left_y},
            {"id": "EyeRightX", "value": data.eye_right_x},
            {"id": "EyeRightY", "value": data.eye_right_y},

            # ======================
            # 눈썹
            # ======================
            {"id": "Brows",       "value": brows},
            {"id": "BrowLeftY",   "value": data.brow_left_y},
            {"id": "BrowRightY",  "value": data.brow_right_y},

            # ======================
            # 입 & 표정
            # ======================
            {"id": "MouthOpen",   "value": data.mouth_open},
            {"id": "MouthSmile",  "value": data.mouth_smile},
            {"id": "CheekPuff",   "value": data.cheek_puff},
            {"id": "TongueOut",   "value": data.tongue_out},
        ]

        await self._send(
            "InjectParameterDataRequest",
            {
                "faceFound": True,
                "mode": "set",
                "parameterValues": params
            }
        )


# ============================================================
# 메인 애플리케이션
# ============================================================
class FaceTrackingApp:
    def __init__(self):
        self.calculator = BlendShapeCalculator()
        self.vts = VTubeStudioAPI()
        self.running = False
        self.data_queue = Queue()
        self.token_file = "vts_token.txt"
    
    def _load_token(self) -> Optional[str]:
        try:
            with open(self.token_file, 'r') as f:
                return f.read().strip()
        except:
            return None
    
    def _save_token(self, token: str):
        with open(self.token_file, 'w') as f:
            f.write(token)
    
    async def _vts_loop(self):
        """VTube Studio 통신 루프"""
        if not await self.vts.connect():
            print("VTube Studio 연결 실패 - 트래킹만 실행합니다")
            return
        
        token = self._load_token()
        if await self.vts.authenticate(token):
            if self.vts.auth_token:
                self._save_token(self.vts.auth_token)
        else:
            return
        
        while self.running:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    await self.vts.send_tracking_data(data)
                await asyncio.sleep(0.016)  # ~60fps
            except Exception as e:
                print(f"VTS 오류: {e}")
                break
        
        await self.vts.close()
    
    def _camera_loop(self):
        """카메라 캡처 루프"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("✗ 카메라를 열 수 없습니다")
            self.running = False
            return
        
        print("✓ 카메라 시작")
        print("\n[조작법]")
        print("  q: 종료")
        print("  d: 디버그 정보 토글")
        print("  e: 눈 표시기 토글")
        print("  s: 스무딩 조절 (+0.1)")
        print("  b: 모든 BlendShape 출력")
        
        show_debug = True
        show_eyes = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 거울 모드
            frame = cv2.flip(frame, 1)
            
            # BlendShape 계산
            data = self.calculator.process_frame(frame)
            
            if data:
                # VTS로 전송
                if self.data_queue.qsize() < 3:
                    self.data_queue.put(data)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("MediaPipe Face Tracking (q: quit)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.calculator.smoothing_factor = min(1.0, 
                    self.calculator.smoothing_factor + 0.1)
                if self.calculator.smoothing_factor > 0.95:
                    self.calculator.smoothing_factor = 0.1
                print(f"스무딩: {self.calculator.smoothing_factor:.1f}")
            elif key == ord('b'):
                # 모든 BlendShape 출력
                bs = self.calculator.get_raw_blendshapes(frame)
                print("\n=== All BlendShapes ===")
                for name, value in sorted(bs.items()):
                    if value > 0.01:
                        print(f"  {name}: {value:.3f}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.calculator.close()
    
    def run(self):
        """애플리케이션 실행"""
        print("=" * 50)
        print("  MediaPipe Face Tracker → VTube Studio")
        print("=" * 50)
        
        self.running = True
        
        # VTS 통신 스레드
        def run_vts():
            asyncio.run(self._vts_loop())
        
        vts_thread = threading.Thread(target=run_vts, daemon=True)
        vts_thread.start()
        
        # 카메라 루프 (메인 스레드)
        self._camera_loop()
        
        self.running = False
        print("\n프로그램 종료")

# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    import sys
    FaceTrackingApp().run()
