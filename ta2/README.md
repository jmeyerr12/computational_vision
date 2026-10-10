# Calibração de câmera com OpenCV

## Uso

1. Crie um ambiente virtual e instale `pip install -r requirements.txt`.
2. Coloque 10 fotos de um tabuleiro na pasta `images/`, com ângulos variados, mesma câmera e mesma resolução.
3. No `main.py`, ajuste `PATTERN` (número de **cantos internos**, em colunas e linhas) e `SQUARE_MM` (lado do quadrado em milímetros).
4. Execute `python3 main.py`.
5. Consulte os resultados na pasta `output/`.

São gerados: matriz intrínseca e distorção (`calibration.npz`), imagens com cantos, imagens sem distorção, comparações, projeções 3D para 2D e tabela CSV com erros em pixels.

## Experimento

O plano Z=0 corresponde ao próprio tabuleiro. Os pontos 3D são escolhidos em cantos cujas posições reais são conhecidas. A posição 2D projetada com `cv.projectPoints` é comparada com o canto observado por `cv.findChessboardCorners`.

**Observação:** os pontos de teste pertencem às imagens usadas na calibração. Portanto, o erro é de ajuste (reprojeção nos dados de treinamento), não avaliação independente. Para uma validação independente, reserve algumas fotos e estime a pose com `cv.solvePnP` usando parte dos cantos, avaliando nos demais.

Documentação: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
