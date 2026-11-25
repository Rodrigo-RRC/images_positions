import mediapipe as mp
import math
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker # o detector, criado a partir das opções.
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions # a configuração. Diz onde está o modelo, se a entrada é imagem ou vídeo, quantos rostos procurar etc.
VisionRunningMode = mp.tasks.vision.RunningMode #  define como uma tarefa de visão computacional processa os dados de entrada
'''
   RunningMode define como uma tarefa de visão computacional processa os dados de entrada. Existem três modos de execução:
    • IMAGE: Processa imagens estáticas e independentes de forma síncrona, sem manter rastreamento entre chamadas.
    • VIDEO: Processa quadros decodificados de um vídeo em ordem, com timestamps crescentes e rastreamento temporal.
    • LIVE_STREAM: Processa fluxos contínuos (como uma câmera) de forma assíncrona com callback, priorizando baixa latência e descartando quadros quando ocupado
'''

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_facial_transformation_matrixes=True,                                                                               
)

with FaceLandmarker.create_from_options(options) as landmarker:
    for caminho in sorted(Path("data").glob("*.jpg")):
        mp_image = mp.Image.create_from_file(str(caminho))
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            print(f"{caminho.name}: REPROVADA: nenhum rosto encontrado na imagem.")
            continue

        r = result.facial_transformation_matrixes[0][:3, :3]

        # MediaPipe usa eixos OpenGL: pitch e roll vêm com sinal oposto à convenção do projeto
        pitch = -math.degrees(math.atan2(r[2, 1], r[2, 2]))
        yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])))
        roll = -math.degrees(math.atan2(r[1, 0], r[0, 0]))

        print(f"{caminho.name}: pitch={pitch:.1f}°  yaw={yaw:.1f}°  roll={roll:.1f}°")
        

