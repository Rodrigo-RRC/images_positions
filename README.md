# images_positions

Validação de posicionamento de faces em imagens a partir dos ângulos **pitch**, **yaw** e **roll** da cabeça.

> Projeto de portfólio construído passo a passo. Este README registra cada etapa e as decisões tomadas.

## Objetivo

Receber uma imagem com um rosto, calcular a orientação da cabeça e **aprovar ou reprovar** a imagem conforme um limite de tolerância (*threshold*) para cada ângulo, funcionando como um *guardrail* de qualidade facial.

```text
imagem ──► detectar rosto ──► pitch / yaw / roll ──► comparar com limites ──► APROVADA / REPROVADA
```

<p align="center">
  <img src="docs/img/pitch-yaw-roll.svg" alt="Pitch: cabeça para baixo (−) ou para cima (+). Yaw: rosto para a esquerda (−) ou para a direita (+). Roll: cabeça inclinada para a esquerda (−) ou para a direita (+)." width="720">
</p>

Rosto frontal: **pitch = yaw = roll = 0°**.

**Convenção de sinais** (esquerda e direita do ponto de vista de quem olha a imagem):

| Ângulo | Negativo (−) | Positivo (+) |
|--------|--------------|--------------|
| **Pitch** | cabeça para baixo | cabeça para cima |
| **Yaw**   | rosto virado para a esquerda | rosto virado para a direita |
| **Roll**  | cabeça inclinada para a esquerda | cabeça inclinada para a direita |

> Cada biblioteca e cada dataset pode usar uma convenção diferente. Ao comparar resultados, é preciso converter os sinais para esta convenção.

---

## Passo 1: Ambiente

Requer Python 3.10+ (desenvolvido com Python 3.14 no Ubuntu 26.04 via WSL).

### 1.1 Dependência de sistema: `libgles2`

O MediaPipe não é só Python: por baixo, ele carrega uma biblioteca compilada em C++ que usa **OpenGL ES** e depende do arquivo `libGLESv2.so.2` do sistema.
Instalações enxutas do Ubuntu, como a do WSL, não trazem esse pacote, e o `pip` não instala dependências de sistema.

```bash
sudo apt update
sudo apt install -y libgles2
```

Sem ele, o erro aparece ao criar o detector:

```text
OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
```

> Se o `apt` reclamar de `dpkg was interrupted` ou `Unmet dependencies`, alguma instalação anterior foi interrompida. Resolva antes com:
> `sudo dpkg --configure -a` e depois `sudo apt --fix-broken install -y`.

### 1.2 Ambiente virtual e pacotes Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mediapipe
```

O `mediapipe` já instala as dependências de que o projeto precisa, entre elas `numpy` e `opencv-contrib-python`.

> ⚠️ **Não instale `opencv-python` separadamente.** Ele disputa o mesmo `import cv2` com o `opencv-contrib-python` trazido pelo MediaPipe e causa conflitos.

## Passo 2: Modelo de detecção facial

Usamos a API **MediaPipe Tasks** (`mediapipe.tasks.vision.FaceLandmarker`).
A API antiga (`mp.solutions.face_mesh`) foi descontinuada e **não existe mais** a partir do MediaPipe 1.x.

A API Tasks exige baixar o modelo separadamente:

```bash
mkdir -p models
wget -O models/face_landmarker.task https://huggingface.co/litert-community/MediaPipe-FaceLandmarker/resolve/main/face_landmarker.task
```

O arquivo `face_landmarker.task` (3,6 MB) é um *bundle* com três modelos em sequência:

```text
imagem ──► detector de rosto (BlazeFace) ──► 478 pontos 3D do rosto ──► expressões (52 blendshapes) + matriz de pose 4x4
```

**Sobre o link:** a documentação oficial passou a hospedar o modelo no Hugging Face (`litert-community`).
O endereço antigo (`storage.googleapis.com/mediapipe-models/...`), comum em tutoriais, ainda funciona, mas pode sair do ar.
Antes de usar, confira sempre o link na [documentação do Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
Os dois endereços entregam o mesmo arquivo:

```text
MD5: b0e7274907a1644404fef66b28dd6d85
```

Verificação: `md5sum models/face_landmarker.task`

## Passo 3: Detectar os pontos do rosto

Script: [`scripts/01_detectar_landmarks.py`](scripts/01_detectar_landmarks.py). Execute **a partir da raiz do projeto**, porque os caminhos são relativos:

```bash
python scripts/01_detectar_landmarks.py
```

Saída:

```text
Rostos encontrados: 1
Pontos no rosto: 478
Ponto 1 (ponta do nariz): NormalizedLandmark(x=0.503, y=0.628, z=-0.121, ...)
```

O resultado (`FaceLandmarkerResult`) organiza os pontos como **lista de rostos → lista de pontos**:

```text
result.face_landmarks          → todos os rostos encontrados
result.face_landmarks[0]       → os 478 pontos do 1º rosto
result.face_landmarks[0][1]    → o ponto de índice 1 (ponta do nariz) do 1º rosto
```

Cada ponto tem coordenadas **normalizadas**:
- `x` e `y` vão de 0 a 1, em proporção à largura e à altura da imagem (`x = 0.5` é o meio da foto);
- `z` é a profundidade relativa: quanto mais negativo, mais perto da câmera.

> As linhas `WARNING`, `W0000` e `INFO` na saída são logs internos do MediaPipe, não erros.

### Imagem sem rosto: primeira regra do guardrail

Se não houver rosto, `result.face_landmarks` vem como lista vazia `[]`, e acessar `[0]` quebraria o programa com `IndexError`.
O script verifica antes e **reprova a imagem com o motivo**:

```python
if not result.face_landmarks:          # lista vazia conta como falso em Python
    print("REPROVADA: nenhum rosto encontrado na imagem.")
```

| Imagem de teste | Resultado |
|-----------------|-----------|
| com um rosto    | 478 pontos detectados |
| sem rosto       | `REPROVADA: nenhum rosto encontrado na imagem.` |

## Passo 4: Calcular pitch, yaw e roll *(em andamento)*

Script: `scripts/02_calcular_angulos.py`.

Para obter a pose, ativamos uma opção nova no detector:

```python
output_facial_transformation_matrixes=True
```

Com ela, além dos 478 pontos, o modelo devolve uma **matriz de pose 4x4** por rosto (`result.facial_transformation_matrixes`).
A matriz diz quanto um rosto 3D padrão teve que ser **girado** e **deslocado** para coincidir com o rosto da foto:

```text
┌─ rotação (3x3) ─────┐ ┌ translação ┐
[ 1.000   0.007  -0.022 │   0.211 ]   x
[-0.011   0.974  -0.224 │  -1.834 ]   y
[ 0.020   0.225   0.974 │ -34.374 ]   z: ~34 cm da câmera
[ 0       0       0     │   1     ]   linha fixa
```
<sub>Matriz da `image_01.jpg`, impressa pelo script.</sub>

### A matriz de rotação, em desenho

👉 **[Abrir a animação interativa](https://claude.ai/artifact/RApU88BF746By99rL7knf4)** (também em [`docs/matriz-rotacao.html`](docs/matriz-rotacao.html): baixe o arquivo e abra no navegador)

Uma cabeça em 3D gira na tela e a matriz muda ao vivo, em 8 passos. A ideia central:

- Três setas ficam presas na cabeça: **vermelha (x)**, no lado; **verde (y)**, no topo da cabeça; e **azul (z)**, no nariz.
- **Cada coluna** da matriz é uma seta. **Cada linha** pergunta quanto essa seta aponta para a direita, para cima ou para a câmera.
- Em cada giro, **uma seta fica parada** (é o eixo do giro) e as outras duas se inclinam:

| Movimento | Seta parada | Casas que mudam |
|-----------|-------------|-----------------|
| **Pitch** (fazer "sim") | vermelha (x) | [cima, azul (z)] e [câmera, verde (y)] |
| **Yaw** (fazer "não") | verde (y) | [direita, azul (z)] e [câmera, vermelha (x)] |
| **Roll** (orelha no ombro) | azul (z) | [direita, verde (y)] e [cima, vermelha (x)] |

Na `image_01.jpg`, só as casas do pitch fogem de 0 e 1 (−0.224 e 0.225): a cabeça está cerca de **13° para baixo**.

---

## Estrutura

```text
images_positions/
├── data/       # imagens de teste (não versionado)
├── docs/       # ilustrações e a animação da matriz de rotação
├── models/     # face_landmarker.task (não versionado)
└── scripts/    # scripts numerados, um por etapa
```

## Roteiro

- [x] Ambiente Python e dependências
- [x] Download e verificação do modelo
- [x] Detectar o rosto e os pontos faciais (*landmarks*)
- [ ] Calcular pitch, yaw e roll
- [ ] Definir os limites por ângulo e validar (guardrail)
- [ ] Medir a precisão contra dataset com ângulos reais (MAE / RMSE)
- [ ] Interface para o usuário enviar imagens
- [ ] Suporte a vídeo
- [ ] Testes automatizados

## Referências

- [MediaPipe Face Landmarker: visão geral](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe Face Landmarker: guia Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [MediaPipe API Python: `FaceLandmarkerResult`](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult)

## Autor

Rodrigo Ribeiro Carvalho
