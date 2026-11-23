import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker # o detector, criado a partir das opções.
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions # a configuração. Diz onde está o modelo, se a entrada é imagem ou vídeo, quantos rostos procurar etc.
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

with FaceLandmarker.create_from_options(options) as landmarker:
    # Synchronous single-image call (stateless across images)
    mp_image = mp.Image.create_from_file("data/image_01.jpg")
    result = landmarker.detect(mp_image)
    print("Rostos encontrados:", len(result.face_landmarks))
    print("Pontos no primeiro rosto:", len(result.face_landmarks[0]))
    print("Ponto 1 (ponta do nariz):", result.face_landmarks[0][1])
   
    