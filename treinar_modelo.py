import os
import pickle
import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from helpers import resize_to_fit

dados = []
rotulos = []
pasta_base_imagens = "base_letras"

imagens = paths.list_images(pasta_base_imagens)

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[1]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # padronizar em 20x20
    imagem = resize_to_fit(imagem, 20, 20)

    # adicionar dimensao para trabalhar com o Keras
    imagem = np.expand_dims(imagem, axis=2)

    # adicionar as listas de dados e rótulos
    rotulos.append(rotulo)
    dados.append(imagem)

rotulos = np.array(rotulos)
dados = np.array(dados, dtype="float") / 255

# separacao em dados de treino e teste
(X_train, X_teste, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# converter com one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o labelbinarizer em um arquivo com pickle
with open('rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# criar e treinar a inteligencia artificial
modelo = Sequential()

# criar as camadas da rede neural
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar a 2 camada da rede neural
modelo.add(Conv2D(50, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# outra camada da rede neural
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))

# camada de saída
# TODO se o captcha tiver numeros trocar o primeiro parametro de 26 para 36
modelo.add(Dense(26, activation="softmax"))

# compilar todas as camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# treinar a inteligencia artificial
# TODO se o captcha tiver numeros trocar o parametro batch_size de 26 para 36
modelo.fit(X_train, Y_train, validation_data=(X_teste, Y_test), batch_size=26, epochs=10, verbose=1)

# salvar o modelo em um arquivo
modelo.save("modelo_treinado.hdf5")
