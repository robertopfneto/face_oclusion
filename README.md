# Oclusão Facial

Detecta, numa foto ou ao vivo pela webcam, **o que está cobrindo o rosto**:

- **regiões cobertas**: olhos, sobrancelhas, nariz, boca, testa e contorno, com o motivo (acessório, roupa, cabelo, mão ou olho fechado);
- **acessórios**: chapéu ou boné, óculos (de grau ou escuros) e máscara;
- a **confiança** de cada decisão.

Roda **inteiro no navegador**, no computador ou no celular, e a imagem não sai do aparelho. A mesma análise existe em **Python** (`oclusao_py/`), para usar num backend.

---

## Sumário

1. [Como funciona](#como-funciona)
2. [Tutorial](#tutorial)
   - [1. Rodar o app no computador](#1-rodar-o-app-no-computador)
   - [2. Usar o app](#2-usar-o-app)
   - [3. Usar no celular](#3-usar-no-celular)
   - [4. Testar e gerar o relatório](#4-testar-e-gerar-o-relatório)
   - [5. Usar em Python](#5-usar-em-python)
   - [6. (Opcional) Servidor PyTorch para comparar](#6-opcional-servidor-pytorch-para-comparar)
3. [Formato da saída (JSON)](#formato-da-saída-json)
4. [Ajustar limiares](#ajustar-limiares)
5. [Estrutura do projeto](#estrutura-do-projeto)
6. [Testes](#testes)
7. [Limitações conhecidas](#limitações-conhecidas)
8. [Modelos e licenças](#modelos-e-licenças)

---

## Como funciona

Três modelos de IA, cada um num *worker* separado para a tela não travar, e regras simples em cima dos resultados deles:

```
câmera / foto ─┬─► 1. FaceLandmarker   478 pontos do rosto + piscada     (todo quadro)
               ├─► 2. Segmentador      classe de cada pixel               (em paralelo)
               └─► 3. Classif. óculos  grau / escuros                     (até ~6x/s)
                                │
                                ▼
          regras (occlusion.js  ⇄  oclusao_py/regras.py)
          regiões · chapéu/boné · máscara · óculos · confiança
                                │
                                ▼
          pontos na tela · etiquetas · painel · JSON
```

### 1. Landmarks: onde está o rosto

O **FaceLandmarker** do MediaPipe acha **478 pontos** no rosto (contorno, olhos, íris, sobrancelhas, nariz e lábios). Também devolve a **piscada** de cada olho (`eyeBlinkLeft`/`Right`). Todo o resto usa esses pontos para saber **onde olhar**.

### 2. Segmentador: o que tem em cada lugar

O **Selfie Multiclass** do MediaPipe classifica cada pixel como **fundo, cabelo, pele do corpo, pele do rosto, roupa ou acessório**. Cada ponto do rosto "lê" a classe embaixo dele. Nos olhos, a leitura é feita num anel um pouco para fora, onde ficam lente e armação.

### 3. As decisões


| O quê               | Como decide                                                                                                                                        | Limiar                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Região coberta**  | fração dos pontos da região que não caíram em "pele do rosto", descontada a *referência*                                                      | 0,40                                             |
| **Olho fechado**     | a **piscada** do MediaPipe **e** a **abertura** da pálpebra (EAR) precisam concordar. Assim, olho pequeno ou semiaberto não conta                 | piscada 0,35→0,70; EAR 0,20→0,10               |
| **Chapéu ou boné** | faixa de 21 pontos **acima da testa** (acompanha a inclinação da cabeça): fração lida como acessório ou roupa. Cabelo não conta              | 0,35                                             |
| **Máscara**         | pontos da **metade de baixo do rosto** (lábios, base do nariz, queixo): fração lida como acessório ou roupa. Mão não conta                    | 0,40                                             |
| **Óculos**          | classificador próprio ([glasses-detector](https://github.com/mantasu/glasses-detector)) sobre o recorte do rosto; dois scores, `grau` e `escuros` | foto 0,65; ao vivo liga em 0,75 e desliga em 0,5 |

**No ao vivo**, chapéu/boné, máscara e óculos passam por **média móvel** (um quadro ruim isolado não muda o resultado) e **histerese** (para ligar e para desligar é preciso passar do limiar com folga, então o resultado não pisca).

**Confiança** é a distância do score ao limiar: 0% em cima do limiar, 100% no extremo. **Alta** ≥ 60%, **média** ≥ 30%, **baixa** abaixo disso. Não é uma probabilidade calibrada, mas mostra onde o sistema está em dúvida.

---

## Tutorial

### 1. Rodar o app no computador

**Pré-requisito:** Python 3 (só para servir os arquivos) e um navegador moderno (Chrome, Edge, Firefox ou Safari).

**Subir.** Na pasta do projeto:

```bash
python3 -m http.server 8765
```

Deixe esse terminal aberto enquanto usa o app. Depois abra um destes endereços:

| Endereço | Para quê |
| --- | --- |
| **http://localhost:8765** | o app |
| **http://localhost:8765/?teste** | o app com a barra de testes: gravar cenários e baixar o relatório ([passo 4](#4-testar-e-gerar-o-relatório)) |
| **http://IP-DO-COMPUTADOR:8765** | outro aparelho na mesma Wi-Fi; só **Enviar foto** funciona, porque o **Ao vivo** exige HTTPS ou localhost ([passo 3](#3-usar-no-celular)) |

Para descobrir o IP do computador: `hostname -I` (Linux), `ipconfig getifaddr en0` (macOS) ou `ipconfig` (Windows).

Na primeira vez, o app baixa os modelos (uns 55 MB) e mostra "Tire uma foto do rosto".

**Parar.** Aperte **Ctrl + C** no terminal do servidor. Se o terminal foi fechado ou o servidor ficou em segundo plano:

```bash
pkill -f "http.server 8765"
```

**"Address already in use"** ao subir: já existe um servidor nessa porta. Use o que já está no ar, pare-o com o comando acima ou suba em outra porta (`python3 -m http.server 8766`).

**Com a comparação do servidor PyTorch** (seção "Servidor" no painel), suba este no lugar do anterior. Ele serve o app e a API juntos, em **http://localhost:8000**. A instalação está no [passo 6](#6-opcional-servidor-pytorch-para-comparar).

```bash
server/.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
# parar: Ctrl + C, ou  pkill -f "uvicorn server.app:app"
```

> **Não abra o `index.html` com dois cliques.** Aberto como arquivo (`file://`), o navegador bloqueia os módulos e os modelos, e o app mostra "Abra por um servidor".

### 2. Usar o app


| Botão                    | O que faz                                                             |
| ------------------------- | --------------------------------------------------------------------- |
| **Ao vivo**               | liga a webcam e analisa cada quadro. Vira **Parar**                    |
| **Capturar**              | aparece durante o ao vivo: congela o quadro atual e analisa como foto |
| **Enviar foto**           | analisa uma imagem do aparelho                                        |
| **Landmarks**             | mostra ou esconde os pontos                                           |
| **Usar como referência** | salva a foto atual como o seu "rosto livre" (veja abaixo)             |

**O que aparece na tela:**

- **Pontos**: verdes nas regiões livres, vermelhos nas cobertas. **Quadrados** acima da testa marcam onde o boné é procurado: amarelos são pontos lidos, laranja são pontos que contaram como boné.
- **Etiquetas vermelhas** acima dos botões: "Chapéu ou boné", "Óculos de grau" / "Óculos escuros" e "Máscara", com a confiança.
- **Painel**: ao lado da imagem no computador; no topo e recolhido no celular (toque para abrir). Cada item tem um selo (verde = livre, vermelho = detectado, cinza = analisando), a confiança e uma barra com o score (o traço é o limiar). Nos olhos, mostra também `abertura` e `piscada`.
- **JSON**, dentro do painel: toque para alternar entre a forma compacta e a completa.
- **Linha de FPS**: quadros por segundo, máscaras por segundo, consultas de óculos por segundo, e se cada modelo roda em GPU ou CPU.

**Dica: use a referência.** Com o rosto livre (sem óculos, boné ou máscara) e de frente, toque em **Usar como referência**. Isso ensina ao app o que é "normal" em cada ponto do **seu** rosto e elimina falsos positivos em lábios e sobrancelhas. A referência fica salva no navegador.

### 3. Usar no celular

O **ao vivo** só funciona em **HTTPS** ou em **localhost**. Essa é uma regra do navegador para liberar a câmera. Opções:


| Opção                                    | Como                                                                                                                                                                                                                 | Ao vivo?                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Mesma Wi-Fi**                            | no celular, abra `http://IP-DO-COMPUTADOR:8765`                                                                                                                                                                       | não, só **Enviar foto** (que também abre a câmera) |
| **Tailscale**                              | com o app Tailscale no celular e no PC: `tailscale serve --bg 8765` no PC e abrir o endereço `https://….ts.net` que ele mostra. Só aparelhos da sua rede acessam. Para desfazer: `tailscale serve --https=443 off` | sim                                                   |
| **Servidor no próprio celular (Android)** | no Termux, dentro da pasta do projeto: `python -m http.server 8765` e abrir `http://localhost:8765`                                                                                                                   | sim                                                   |

Para o celular, a pasta precisa de: `index.html`, `app.js`, `occlusion.js`, `seg-worker.js`, `oculos-worker.js`, `vendor/` e `models/` (~65 MB).

### 4. Testar e gerar o relatório

O **modo de teste** mede o app no seu aparelho, com o seu rosto, e gera um relatório para calibrar os limiares.

1. Abra o app com `?teste` no fim do endereço: **http://localhost:8765/?teste**
2. Aparece uma barra extra. Escolha o **cenário**: sem nada, chapéu ou boné, gorro/capuz, óculos de grau, óculos escuros, máscara, mão na boca, franja, olho pequeno/semiaberto, olhos fechados ou um olho fechado.
3. Toque em **Ao vivo** e depois em **Gravar 5 s**. Fique no cenário, de frente, a uns 50 cm da câmera. São cerca de 25 amostras. (Com uma foto, cada toque grava 1 amostra.)
4. Repita para os outros cenários. Vale repetir com pouca luz e com a cabeça inclinada.
5. Toque em **Baixar relatório**.

O relatório é um JSON com, para cada amostra: scores e confiança de cada região e acessório, abertura e piscada dos olhos, **as classes que o segmentador leu** na faixa do boné e na metade de baixo do rosto, e o FPS. No fim vem um **`resumo`** por cenário (média, mínimo, máximo e quantas vezes cada item foi detectado). **Nenhuma imagem vai no arquivo.** As amostras ficam na memória até você recarregar a página.

### 5. Usar em Python

A mesma análise, sem navegador: **mesmos modelos, mesmas regras, mesmo JSON**. Os testes garantem que as regras respondem igual ao app, e os landmarks saem idênticos.

**Instalar** (testado em Python 3.12 e 3.14):

```bash
conda create -n oclusao python=3.14
conda activate oclusao
pip install -r requirements.txt
python -m pytest tests/python -q          # deve dar "41 passed"
```

Sem conda: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

**Pela linha de comando** (rode na raiz do projeto):

```bash
python -m oclusao_py.pipeline foto.jpg --compacto
python -m oclusao_py.pipeline foto1.jpg foto2.jpg          # forma completa, uma linha JSON por foto
python -m oclusao_py.pipeline foto.jpg --referencia rosto_livre.jpg   # com referência (veja abaixo)
```

**Como biblioteca: fotos**

```python
import numpy as np
from PIL import Image
from oclusao_py.modelos import Modelos
from oclusao_py.pipeline import analisar, to_compacto

with Modelos() as m:                                   # carrega os 3 modelos uma vez
    rgb = np.asarray(Image.open("foto.jpg").convert("RGB"))
    resultado = analisar(rgb, m)                       # forma completa
    print(to_compacto(resultado)["acessorios"])        # {'chapeu_ou_bone': False, 'oculos': 'grau', 'mascara': False}
```

O fluxo dentro de `analisar()` segue as etapas, nesta ordem (cada uma é uma função em `pipeline.py`):

```
achar_rosto → ler_classes → decidir_regioes → decidir_chapeu_e_mascara → decidir_oculos → montar_saida
```

Extras:

- **Referência** (o "Usar como referência" do app): `ref = criar_referencia(rgb_rosto_livre, m)` e depois `analisar(rgb, m, referencia=ref)`. Use uma foto do rosto livre e de frente.
- **Etiquetas**: `etiquetas(resultado)` devolve os acessórios detectados com o texto do app, por exemplo `[{"text": "Óculos de grau", "confianca": 0.71, ...}]`.
- **Depurar**: `analisar(..., detalhes=True)` inclui landmarks, pontos vermelhos, classes lidas e scores brutos.

**Como biblioteca: vídeo / ao vivo**

`SessaoAoVivo` guarda a média móvel e a histerese entre quadros. **Crie uma por câmera ou cliente**, nunca compartilhe entre fluxos.

```python
from oclusao_py.sessao import SessaoAoVivo

with SessaoAoVivo(segmentar_a_cada=2) as s:            # segmentador a cada 2 quadros (mais leve)
    s.usar_como_referencia(quadro_rosto_livre)         # opcional, como o botão do app
    for quadro_rgb in quadros:                         # numpy uint8 (altura, largura, 3), em RGB
        resultado = s.processar(quadro_rgb)
```

Exemplo com a webcam via OpenCV (precisa de `pip install opencv-python`, que não está no `requirements.txt`; lembre que o OpenCV entrega **BGR**):

```python
import cv2
from oclusao_py.sessao import SessaoAoVivo
from oclusao_py.pipeline import to_compacto

cam = cv2.VideoCapture(0)
with SessaoAoVivo(segmentar_a_cada=2) as s:
    while True:
        ok, bgr = cam.read()
        if not ok:
            break
        r = s.processar(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        print(to_compacto(r).get("acessorios"))
```

**Arquivos do pacote:**


| Arquivo                  | Papel                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `oclusao_py/regras.py` | tradução 1:1 de `occlusion.js`: limiares, regiões, olho fechado, chapéu/boné, máscara, óculos, confiança, etiquetas |
| `oclusao_py/modelos.py` | carrega FaceLandmarker (com piscada), segmentador e `oculos.onnx`; modo `IMAGE` (fotos) ou `VIDEO` (sequência) |
| `oclusao_py/pipeline.py` | o fluxo, etapa por etapa; `analisar()`, `criar_referencia()`, `etiquetas()`, `to_compacto()` e a linha de comando |
| `oclusao_py/sessao.py` | `SessaoAoVivo`: o mesmo fluxo por quadro, com memória entre quadros e `usar_como_referencia()` |

### 6. (Opcional) Servidor PyTorch para comparar

Um servidor com modelos maiores serve de **segunda opinião** para chapéu/boné ([FaRL](https://github.com/FacePerceiver/facer)) e máscara ([SigLIP](https://huggingface.co/prithivMLmods/Face-Mask-Detection)). Com ele ligado, o painel ganha uma seção **Servidor** para comparar com as regras do navegador. **Não é necessário para o app funcionar.** Nesse modo, o quadro é enviado ao servidor.

```bash
uv venv --python 3.12 server/.venv
uv pip install --python server/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
uv pip install --python server/.venv/bin/python -r server/requirements.txt
server/.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Abra **http://localhost:8000**: ele serve o app e a API juntos. Cada consulta leva ~400 ms em CPU. Outros scripts:

- `server/smoke.py`: teste de fumaça com acessórios desenhados;
- `server/export_onnx.py`: regera `models/oculos.onnx` e confere a paridade com o PyTorch;
- `server/eval_oculos.py`: avalia o detector de óculos numa pasta de fotos reais.

**Opções de URL do app:** `?teste` (modo de teste), `?cpu` (força CPU), `?api=http://host:8000/api/` (servidor em outro endereço).

---

## Formato da saída (JSON)

Exemplo real (a foto de teste com óculos escuros desenhados).

**Compacto**, o padrão no painel e no `--compacto`:

```json
{
  "rosto_detectado": true,
  "olho_esquerdo": true,
  "olho_direito": true,
  "sobrancelhas": true,
  "nariz": false,
  "boca": false,
  "testa": true,
  "contorno": false,
  "acessorios": {
    "chapeu_ou_bone": false,
    "oculos": "escuros",
    "mascara": false
  }
}
```

**Completo** (trecho):

```json
{
  "rosto_detectado": true,
  "olho_esquerdo": { "ocluso": true, "motivo": "acessorio", "score": 1.0, "confianca": 1.0, "abertura": 0.345, "piscada": 0.08 },
  "boca": { "ocluso": false, "motivo": null, "score": 0.0, "confianca": 1.0 },
  "acessorios": {
    "chapeu_ou_bone": { "presente": false, "score": 0.0, "confianca": 1.0 },
    "oculos": { "presente": true, "tipo": "escuros", "grau": 0.0, "escuros": 0.99, "confianca": 0.98 },
    "mascara": { "presente": false, "score": 0.0, "confianca": 1.0 }
  }
}
```


| Campo                      | Valores                                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| `rosto_detectado`          | `false` quando não há rosto; aí as regiões vêm `null` e não há `acessorios`                    |
| região (compacto)         | `true` = coberta                                                                                      |
| `motivo`                   | `acessorio`, `roupa`, `cabelo`, `mao_ou_pele`, `fundo`, `fechado` (olho fechado) ou `null`            |
| `score`                    | fração coberta, de 0 a 1 (nos olhos, o maior entre cobertura e "fechado")                           |
| `abertura`, `piscada`      | só nos olhos: abertura da pálpebra (EAR) e piscada do MediaPipe                                     |
| `oculos` (compacto)        | `"grau"`, `"escuros"` ou `false`                                                                      |
| acessório "não decidido" | `"analisando"` (ainda sem resultado) ou `"fora do quadro"` (chapéu/boné com a faixa fora da imagem) |
| `segmentacao: false`       | o segmentador não carregou: só olho fechado é detectado                                            |
| `servidor`                 | só no app com o servidor PyTorch ligado                                                              |

---

## Ajustar limiares

Os limiares ficam em **dois lugares, que precisam ter os mesmos valores**:

- `occlusion.js`, usado pelo app: `DEFAULTS` (regiões e olhos), `RULES` (chapéu/boné e máscara), `GLASSES` (óculos);
- `oclusao_py/regras.py`, usado pelo Python: os mesmos dicionários, com os mesmos nomes.

Depois de mudar, regere as respostas de referência e rode os testes:

```bash
node tests/golden.mjs                    # grava tests/fixtures/golden.json a partir do occlusion.js
python -m pytest tests/python -q         # o Python precisa responder igual
npm test                                 # testes do JS
```

Se JS e Python divergirem, o teste aponta o campo exato, por exemplo `estado.olho_esquerdo.score: python=0.83 js=1`.

**Para escolher os valores, use o relatório do [modo de teste](#4-testar-e-gerar-o-relatório)**: ele mostra, por cenário, a faixa de scores com e sem o acessório.

---

## Estrutura do projeto

```
index.html            interface (botões, painel, estilos)
app.js                câmera/foto, landmarks, orquestração, desenho, painel, modo de teste
occlusion.js          TODAS as regras e limiares do app (sem navegador; testado)
seg-worker.js         segmentador em Web Worker
oculos-worker.js      classificador de óculos (ONNX) em Web Worker
models/               face_landmarker.task · selfie_multiclass_256x256.tflite · oculos.onnx
vendor/               MediaPipe tasks-vision 0.10.35 e onnxruntime-web 1.30 (servidos localmente)
oclusao_py/           a mesma análise em Python (regras, modelos, pipeline, sessão ao vivo)
server/               servidor PyTorch opcional + exportação/avaliação dos óculos
tests/
  occlusion.test.mjs  testes das regras em JS
  golden.mjs          gera as respostas de referência do JS para o Python
  python/             paridade Python × JS e pipeline com os modelos reais
  fixtures/           landmarks, respostas de referência e fotos de teste
requirements.txt      dependências do Python (oclusao_py + testes)
```

---

## Testes


| Comando                            | O quê                                                      | Tempo |
| ---------------------------------- | ----------------------------------------------------------- | ----- |
| `npm test`                         | regras do JS (`occlusion.js`)                               | < 1 s |
| `python -m pytest tests/python -q` | Python responde igual ao JS + pipeline com os modelos reais | ~6 s  |
| `node tests/golden.mjs`            | regera as respostas de referência (depois de mudar o JS)   | < 1 s |
| modo `?teste` no aparelho           | **validação de verdade**, com a sua câmera e o seu rosto | —    |

Os testes ponta a ponta com navegador automático (`npm run test:e2e`) são lentos e usam um segmentador simulado. Eles não fazem parte da validação, e um deles falha desde o início do projeto: a franja da foto de teste é marcada como testa coberta.

---

## Limitações conhecidas

- **Limiares calibrados com poucos testes.** Os valores atuais vêm de testes manuais de uma pessoa. Use o modo de teste com mais pessoas, câmeras e luzes antes de confiar em produção.
- **Chapéu/boné e máscara dependem do segmentador.** Ele tem uma classe "acessório" genérica e não documenta quais acessórios reconhece. Boné e máscara funcionaram nos testes, mas modelos de cores ou formatos incomuns podem escapar.
- **Óculos de grau com armação fina** são o caso mais difícil para o classificador.
- **Olho fechado** exige que piscada e abertura concordem. Uma piscada rápida ou o rosto muito de lado podem escapar.
- **iPhone**: não testado.
- **Recorte dos óculos no Python**: o redimensionamento é feito com PIL, e o navegador usa `createImageBitmap`. Os scores podem diferir um pouco na segunda casa decimal.

## Modelos e licenças


| Modelo                                                 | Origem                                                                                                              | Licença                          |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| FaceLandmarker (`face_landmarker.task`)                | [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) (Google)                 | ver model card do MediaPipe       |
| Selfie Multiclass (`selfie_multiclass_256x256.tflite`) | [MediaPipe Image Segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter) (Google) | ver model card;**não conferida** |
| Óculos (`oculos.onnx`)                                | [glasses-detector](https://github.com/mantasu/glasses-detector), exportado por `server/export_onnx.py`              | MIT                               |
| onnxruntime-web (`vendor/ort/`)                        | [Microsoft](https://github.com/microsoft/onnxruntime)                                                               | MIT                               |
| FaRL (só no servidor)                                 | [facer](https://github.com/FacePerceiver/facer)                                                                     | MIT                               |
| SigLIP Face-Mask-Detection (só no servidor)           | [prithivMLmods](https://huggingface.co/prithivMLmods/Face-Mask-Detection)                                           | Apache 2.0                        |


Roberto Neto w/ claude xd
