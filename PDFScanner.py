import cv2
import numpy as np
from pdf2image import convert_from_path
from google.colab import userdata
import easyocr
import json
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#import openai

POPPLER_PATH = r"/usr/bin/"

class PDFScanner:
    def __init__(self, pdf_path, dpi=300):
        """
        Inicializa la clase y convierte el PDF en imágenes.
        :param pdf_path: Ruta del PDF escaneado.
        :param dpi: Resolución para convertir las páginas en imágenes.
        """
        self.pdf_path = pdf_path
        self.dpi = dpi
        
        self.imagenes = self.__convertir_pdf_a_imagenes()



    def __convertir_pdf_a_imagenes(self):
        """Convierte las páginas del PDF en imágenes y las almacena en una lista."""
        imagenes_pil = convert_from_path(
            self.pdf_path, dpi=self.dpi, poppler_path=POPPLER_PATH
        )
        
        imagenes_cv = []
        for imagen_pil in imagenes_pil:
            imagen_cv = np.array(imagen_pil)
            if len(imagen_cv.shape) == 3:
                imagen_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises
            imagenes_cv.append(imagen_cv)
        
        return imagenes_cv
    def mostrar_pagina(self,numero_de_pagina=1,figsize = (10,10)):

        plt.figure(figsize =figsize)  

        plt.imshow(self.imagenes[numero_de_pagina], cmap=cm.gray)


        plt.axis('off')

        plt.show()

    def aplicar_threshold(self):
        """Aplica threshold binario con Otsu a todas las imágenes."""
        self.imagenes = [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for img in self.imagenes]

    def aplicar_medianblur(self, kernel_size=5):
        """Aplica filtro de mediana para reducir ruido."""
        self.imagenes = [cv2.medianBlur(img, kernel_size) for img in self.imagenes]

    def rotar_90_grados(self, indice):
        """
        Rota la imagen en el índice especificado 90 grados en sentido antihorario.
        :param indice: Índice de la imagen en la lista self.imagenes.
        """
        if 0 <= indice < len(self.imagenes):
            self.imagenes[indice] = cv2.rotate(self.imagenes[indice], cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"Índice {indice} fuera de rango. Hay {len(self.imagenes)} imágenes.")

    def extraer_texto(self):
        """Extrae el texto de todas las imágenes usando EasyOCR."""
        self.reader = easyocr.Reader(['es'])  # Soporte para español e inglés
        texto_total = ""
        for img in self.imagenes:
            resultados = self.reader.readtext(img, detail=0)  # detail=0 solo devuelve el texto
            texto_total += "\n".join(resultados) + "\n"
        return texto_total
    '''
    def extraer_datos_con_ai(self,prompt,funcion,dataframe=True):
        openai.api_key = userdata.get('api_key')
        """
        Usa la API de OpenAI para extraer los datos clave del texto.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": funcion},
                    {"role": "user", "content": prompt}]
        )
        response = response["choices"][0]["message"]["content"]

        if dataframe:

          response = json.loads(response)
          return pd.DataFrame([response])
        else:
          return response
    '''
       
