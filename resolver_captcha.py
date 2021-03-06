import glob

from tensorflow.keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from tratar_captcha import tratar_imagens


def quebrar_captcha():
    with open("rotulos_modelo.dat", "rb") as arquivo_tradutor:
        lb = pickle.load(arquivo_tradutor)

    modelo = load_model("modelo_treinado.hdf5")

    tratar_imagens('resolver', pasta_destino='resolver')

    arquivos = list(paths.list_images("resolver"))
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

        # encontrar os contornos de cada letra
        contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filtrar os contornos que são realmente letras
        regiao_letras = []
        for contorno in contornos:
            (x, y, largura, altura) = cv2.boundingRect(contorno)
            area = cv2.contourArea(contorno)
            if area > 115:
                regiao_letras.append((x, y, largura, altura))

        regiao_letras = sorted(regiao_letras, key=lambda x: x[0])

        imagem_final = cv2.merge([imagem] * 3)
        previsao = []

        for i, retangulo in enumerate(regiao_letras, start=1):
            x, y, largura, altura = retangulo
            imagem_letra = imagem[y - 2:y + altura + 2, x - 2:x + largura + 2]
            imagem_letra = resize_to_fit(imagem_letra, 20, 20)

            # tratamento pro keras funcionar
            imagem_letra = np.expand_dims(imagem_letra, axis=2)
            imagem_letra = np.expand_dims(imagem_letra, axis=0)

            letra_prevista = modelo.predict(imagem_letra)
            letra_prevista = lb.inverse_transform(letra_prevista)[0]
            previsao.append(letra_prevista)

        print("".join(previsao))
        # return "".join(previsao)


if __name__ == '__main__':
    quebrar_captcha()
