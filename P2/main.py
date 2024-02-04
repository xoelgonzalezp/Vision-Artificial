import os
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.morphology import closing, disk, opening
from skimage.segmentation import morphological_chan_vese


def get_image_path(path):
    image = io.imread(path)
    return image


def plotImage(num_plots, images, titles, image_name):
    folder_path = os.path.join("./resultados", image_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig, axs = plt.subplots(1, num_plots, figsize=(10, 5))
    for i in range(num_plots):
        if num_plots == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    save_path = os.path.join(folder_path, f"{image_name}_{len(os.listdir(folder_path))}.png")
    fig.savefig(save_path)


def store_evolution_in(lst):
    def _store(x):
        lst.append(np.copy(x))

    return _store


def detectar_disco(inImage, image_name):
    inImage2 = inImage.copy()
    canal_verde = inImage[:, :, 1]
    plotImage(2, [inImage2, canal_verde], ['Imagen Original', 'Canal verde'], image_name)

    canal_suavizado = cv2.GaussianBlur(canal_verde, (5, 5), 0)
    kernel = np.ones((3, 3), np.float32) / 9
    filtrado = cv2.filter2D(canal_suavizado, -1, kernel)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(filtrado)
    lado_ventana = 200
    x_inicio = max(maxLoc[0] - lado_ventana // 2, 0)
    y_inicio = max(maxLoc[1] - lado_ventana // 2, 0)
    x_fin = min(x_inicio + lado_ventana, inImage.shape[1])
    y_fin = min(y_inicio + lado_ventana, inImage.shape[0])
    roi = inImage2[y_inicio:y_fin, x_inicio:x_fin]
    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_umbral = cv2.threshold(roi_gris, 1, 255, cv2.THRESH_BINARY)

    M = cv2.moments(roi_umbral)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    centroide = (x_inicio + cX, y_inicio + cY)
    cv2.rectangle(inImage, (x_inicio, y_inicio), (x_fin, y_fin), (0, 255, 0), 2)
    cv2.circle(inImage, centroide, 5, (0, 0, 255), -1)
    plotImage(2, [inImage, roi], ['Ventana y centroide en la imagen original', 'Ventana ROI'], image_name)
    return centroide, roi, x_inicio, y_inicio, x_fin, y_fin, inImage2


def eliminar_vasos(image, image_name):
    canal_rojo = image[:, :, 0]
    canal_rojo_mejorado = cv2.GaussianBlur(canal_rojo, (19, 19), 0)
    canal_mejorado = cv2.addWeighted(canal_rojo_mejorado, 1.5, canal_rojo_mejorado, -0.5, 0)
    plotImage(2, [canal_rojo, canal_mejorado], ['Canal rojo', 'Canal rojo mejorado'], image_name)
    SE = disk(20)
    cierre = closing(canal_mejorado, SE)
    cierre = cierre.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(5, 5))
    cierre = clahe.apply(cierre)
    bordes = canny(cierre, 2.5)
    bordes = (bordes * 255).astype(np.uint8)
    kernel = np.ones((1, 3), np.uint8)
    bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
    plotImage(3, [cierre, bordes, bordes_dilatados],
              ['Vasos sanguíneos eliminados', 'Detección de bordes con Canny', 'Bordes con Canny dilatados'],
              image_name)
    return bordes_dilatados


def segmentacion_disco(image, edges, image_name):
    init = np.zeros_like(edges)
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    r = min(cx, cy) - 15
    cv2.circle(init, (cx, cy), r, 1, -1)
    evol = []
    callback = store_evolution_in(evol)
    snake = morphological_chan_vese(edges, num_iter=66, init_level_set=init, smoothing=1, iter_callback=callback)
    SE = disk(20)
    snake = closing(snake, SE)
    snake_contorno = np.ma.masked_where(snake == 0, snake)
    contornos, _ = cv2.findContours(snake.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imagen_contorno = image.copy()
    cv2.drawContours(imagen_contorno, contornos, -1, (255, 255, 0), 1)

    for contour in contornos:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(imagen_contorno, (cx, cy), 3, (0, 0, 255), -1)

    contornos, _ = cv2.findContours(snake_contorno.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contornos, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(max_contour)
    imagen_contorno_ajustado = image.copy()
    cv2.ellipse(imagen_contorno_ajustado, ellipse, (0, 255, 0), 1)
    mask_disco = np.zeros_like(image, dtype=np.uint8)
    cv2.ellipse(mask_disco, ellipse, (255, 255, 255), -1)
    mascara_disco = cv2.cvtColor(mask_disco, cv2.COLOR_BGR2GRAY)
    plotImage(4, [snake, imagen_contorno, imagen_contorno_ajustado, mascara_disco],
              ['Snake aplicado', 'Contorno detectado', 'Contorno ajustado', 'Máscara final del disco'], image_name)
    return mascara_disco, imagen_contorno_ajustado


def segmentacion_copa(roi, contorno_ajus, image_name):
    if roi.ndim == 3 and roi.shape[2] == 3:
        green_channel = roi[:, :, 1]
    else:
        green_channel = roi

    inverted_green = 255 - green_channel
    inverted_green_mejorado = cv2.GaussianBlur(inverted_green, (5, 5), 0)
    inverted_green_mejorado = cv2.addWeighted(inverted_green_mejorado, 0.0, inverted_green, 2, 0)
    structuring_element = disk(15)
    cierre = opening(inverted_green_mejorado, structuring_element)
    cierre = cierre.astype(np.uint8)
    cierre_umbral = cv2.adaptiveThreshold(cierre, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=33,
                                          C=50)
    cierre_umbral = cv2.bitwise_not(cierre_umbral)
    bordes = canny(cierre_umbral, 2.3, 20, 56)
    bordes = (bordes * 255).astype(np.uint8)
    SE = disk(12)
    bordes_cierre = closing(bordes, SE)
    contornos, _ = cv2.findContours(bordes_cierre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos, _ = cv2.findContours(bordes_cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contorno = max(contornos, key=cv2.contourArea)

    M = cv2.moments(max_contorno)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        distances = [np.sqrt((cx - point[0][0]) ** 2 + (cy - point[0][1]) ** 2) for point in max_contorno]
        max_distance = max(distances)

        if max_distance > 34:
            ellipse = cv2.fitEllipse(max_contorno)
            cv2.ellipse(contorno_ajus, ellipse, (0, 255, 255), 1)
            mask_copa = np.zeros_like(roi)
            cv2.ellipse(mask_copa, ellipse, (255, 255, 255), -1)
        else:
            cv2.circle(contorno_ajus, (cx, cy), 1, (0, 0, 255), -1)
            cv2.circle(contorno_ajus, (cx, cy), int(max_distance), (0, 255, 255), 1)
            mask_copa = np.zeros_like(roi)
            cv2.circle(mask_copa, (cx, cy), int(max_distance), (255, 255, 255), -1)

        mask_copa = cv2.cvtColor(mask_copa, cv2.COLOR_BGR2GRAY)
    else:
        print("No se pudo calcular el centroide del contorno.")

    plotImage(2, [inverted_green, inverted_green_mejorado],
              ['Canal verde invertido : Realce de la copa', 'Zona oscura'], image_name)
    plotImage(4, [cierre, cierre_umbral, bordes, bordes_cierre],
              ['Eliminación huecos vasos', 'Zona umbralizada', 'Bordes detectados', 'Cierre para unir regiones'],
              image_name)
    plotImage(2, [contorno_ajus, mask_copa], ['Contorno de la copa detectado', 'Máscara final de la copa'], image_name)
    return mask_copa


def calcular_cdr(m_disco, m_copa, roi, imagen_or, x_inicio, y_inicio, x_fin, y_fin, truth1, truth2, image_name):
    truth1 = (truth1 > 0).astype(np.uint8)
    truth2 = (truth2 > 0).astype(np.uint8)
    pixeles_1 = np.where(truth1 == 1)
    altura_disco_truth = np.max(pixeles_1[0]) - np.min(pixeles_1[0])
    pixeles_2 = np.where(truth2 == 1)
    altura_copa_truth = np.max(pixeles_2[0]) - np.min(pixeles_2[0])

    valor_referencia = altura_copa_truth / altura_disco_truth if altura_disco_truth > 0 else 0
    valor_referencia = round(valor_referencia, 2)
    print("CDR en ground truth:", valor_referencia)

    puntos_disco = np.where(m_disco == 255)
    altura_max_disco = np.max(puntos_disco[0])
    altura_min_disco = np.min(puntos_disco[0])
    altura_disco = altura_max_disco - altura_min_disco
    puntos_copa = np.where(m_copa == 255)
    altura_max_copa = np.max(puntos_copa[0])
    altura_min_copa = np.min(puntos_copa[0])
    altura_copa = altura_max_copa - altura_min_copa

    valor_predicho = altura_copa / altura_disco if altura_disco > 0 else 0
    valor_predicho = round(valor_predicho, 2)
    print("CDR en el segmentado:", valor_predicho)
    mae = np.abs(valor_predicho - valor_referencia)
    mae = round(mae, 2)
    print("MAE: ", mae)

    imagen_resultante = np.zeros_like(roi, dtype=np.uint8)
    imagen_resultante[m_disco == 255] = [0, 0, 255]
    imagen_resultante[m_copa == 255] = [0, 255, 0]

    plotImage(2, [roi, imagen_resultante], ['ROI', 'ROI con Máscaras de Disco y Copa'], image_name)

    imagen_resultante = np.zeros_like(imagen_or, dtype=np.uint8)
    imagen_resultante[y_inicio:y_fin, x_inicio:x_fin][m_disco == 255] = [0, 0, 255]
    imagen_resultante[y_inicio:y_fin, x_inicio:x_fin][m_copa == 255] = [0, 255, 0]

    imagen_resultante = imagen_or.copy()
    imagen_resultante[y_inicio:y_fin, x_inicio:x_fin][m_disco == 255] = [0, 0, 255]
    imagen_resultante[y_inicio:y_fin, x_inicio:x_fin][m_copa == 255] = [0, 255, 0]

    segmentado_disco = np.zeros_like(imagen_or[:, :, 0])
    segmentado_disco[y_inicio:y_fin, x_inicio:x_fin][m_disco == 255] = 1
    segmentado_copa = np.zeros_like(imagen_or[:, :, 0])
    segmentado_copa[y_inicio:y_fin, x_inicio:x_fin][m_copa == 255] = 1

    plotImage(2, [imagen_or, imagen_resultante], ['Imagen Original', 'Imagen con Máscaras de Disco y Copa'], image_name)

    return mae, imagen_resultante, segmentado_disco, segmentado_copa, valor_predicho, valor_referencia


def calcular_metrica_disco(segmentacion, ground_truth, image_name):
    print("Métricas en el segmentado del disco")
    print("-------------------------------------------")
    ground_truth = (ground_truth > 0).astype(np.uint8)
    segmentacion = (segmentacion > 0).astype(np.uint8)
    plotImage(2, [segmentacion, ground_truth], ['Segmentado manual disco', 'Segmentado Truth disco'], image_name)

    Ss = np.sum(segmentacion)
    St = np.sum(ground_truth)
    VP = np.sum(np.logical_and(segmentacion == 1, ground_truth == 1))
    FP = Ss - VP
    FN = St - VP
    VN = segmentacion.size - (Ss + St - VP)

    print(f"{'VP (Verdaderos Positivos)':<30}{VP:<30}")
    print(f"{'FN (Falsos Negativos)':<30}{FN:<30}\n")
    print(f"{'FP (Falsos Positivos)':<30}{FP:<30}")
    print(f"{'VN (Verdaderos Negativos)':<30}{VN}")
    print("\n")

    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    precision = round(precision, 3)
    sensibilidad = VP / (VP + FN) if (VP + FN) > 0 else 0
    sensibilidad = round(sensibilidad, 3)
    DSC = (2 * VP) / (2 * VP + FP + FN) if (2 * VP + FP + FN) > 0 else 0
    DSC = round(DSC, 3)
    similitud = (1 - (np.sqrt((1 - precision) ** 2 + (1 - sensibilidad) ** 2) / np.sqrt(2)))
    similitud = round(similitud, 3)

    print(f"{'Métrica':<15}{'Valor':<15}")
    print(f"{'-' * 30}")
    print(f"{'Precisión':<15}{precision}")
    print(f"{'Sensibilidad':<15}{sensibilidad}")
    print(f"{'DSC':<15}{DSC}")
    print(f"{'Similitud':<15}{similitud}")
    return precision, sensibilidad, DSC, similitud


def calcular_metrica_copa(segmentacion, ground_truth, image_name):
    print("Métricas en el segmentado de la copa")
    print("-------------------------------------------")
    ground_truth = (ground_truth > 0).astype(np.uint8)
    segmentacion = (segmentacion > 0).astype(np.uint8)
    plotImage(2, [segmentacion, ground_truth], ['Segmentado manual copa', 'Segmentado Truth copa'], image_name)

    Ss = np.sum(segmentacion)
    St = np.sum(ground_truth)
    VP = np.sum(np.logical_and(segmentacion == 1, ground_truth == 1))
    FP = Ss - VP
    FN = St - VP
    VN = segmentacion.size - (Ss + St - VP)

    print(f"{'VP (Verdaderos Positivos)':<30}{VP:<30}")
    print(f"{'FN (Falsos Negativos)':<30}{FN:<30}\n")
    print(f"{'FP (Falsos Positivos)':<30}{FP:<30}")
    print(f"{'VN (Verdaderos Negativos)':<30}{VN}")
    print("\n")

    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    precision = round(precision, 3)
    sensibilidad = VP / (VP + FN) if (VP + FN) > 0 else 0
    sensibilidad = round(sensibilidad, 3)
    DSC = (2 * VP) / (2 * VP + FP + FN) if (2 * VP + FP + FN) > 0 else 0
    DSC = round(DSC, 3)
    similitud = (1 - (np.sqrt((1 - precision) ** 2 + (1 - sensibilidad) ** 2) / np.sqrt(2)))
    similitud = round(similitud, 3)

    print(f"{'Métrica':<15}{'Valor':<15}")
    print(f"{'-' * 30}")
    print(f"{'Precisión':<15}{precision}")
    print(f"{'Sensibilidad':<15}{sensibilidad}")
    print(f"{'DSC':<15}{DSC}")
    print(f"{'Similitud':<15}{similitud}")
    return precision, sensibilidad, DSC, similitud


def procesar_imagenes(image_names, image_dir):
    total_precision_disco = 0
    total_sensibilidad_disco = 0
    total_dsc_disco = 0
    total_similitud_disco = 0
    total_precision_copa = 0
    total_sensibilidad_copa = 0
    total_similitud_copa = 0
    total_dsc_copa = 0
    total_mae = 0
    resultados_cdr = pd.DataFrame(columns=['Retinografía', 'Predicción CDR', 'Ground Truth CDR', 'MAE'])
    resultados_disco = pd.DataFrame(columns=['Retinografía', 'Precisión', 'Sensibilidad', 'DSC', 'Similitud'])
    resultados_copa = pd.DataFrame(columns=['Retinografía', 'Precisión', 'Sensibilidad', 'DSC', 'Similitud'])

    for image_name in image_names:
        image_path = os.path.join(image_dir, f"{image_name}.png")

        print("Imagen: ", image_name)
        truth1_path = os.path.join(image_dir, f"{image_name}_disc.png")
        truth2_path = os.path.join(image_dir, f"{image_name}_cup.png")
        image = io.imread(image_path)
        truth1 = io.imread(truth1_path, cv2.IMREAD_GRAYSCALE)
        truth2 = io.imread(truth2_path, cv2.IMREAD_GRAYSCALE)
        centroide, roi, x_inicio, y_inicio, x_fin, y_fin, imagen2 = detectar_disco(image, image_name)
        bordes = eliminar_vasos(roi, image_name)
        mascara_disco, contorno_aj = segmentacion_disco(roi, bordes, image_name)
        mascara_copa = segmentacion_copa(roi, contorno_aj, image_name)
        mae, _, segmentado_disco, segmentado_copa, valor_predicho, valor_referencia = calcular_cdr(mascara_disco,
                                                                                                   mascara_copa, roi,
                                                                                                   imagen2, x_inicio,
                                                                                                   y_inicio, x_fin,
                                                                                                   y_fin, truth1,
                                                                                                   truth2, image_name)

        precision_disco, sensibilidad_disco, dsc_disco, similitud_disco = calcular_metrica_disco(segmentado_disco,
                                                                                                 truth1, image_name)
        nuevo_resultado_disco = {'Retinografía': image_name, 'Precisión': precision_disco,
                                 'Sensibilidad': sensibilidad_disco, 'DSC': dsc_disco, 'Similitud': similitud_disco}
        resultados_disco = resultados_disco._append(nuevo_resultado_disco, ignore_index=True)

        precision_copa, sensibilidad_copa, dsc_copa, similitud_copa = calcular_metrica_copa(segmentado_copa, truth2,
                                                                                            image_name)
        nuevo_resultado_copa = {'Retinografía': image_name, 'Precisión': precision_copa,
                                'Sensibilidad': sensibilidad_copa, 'DSC': dsc_copa, 'Similitud': similitud_copa}
        resultados_copa = resultados_copa._append(nuevo_resultado_copa, ignore_index=True)

        nuevo_resultado_cdr = {'Retinografía': image_name, 'Predicción CDR': valor_predicho,
                               'Ground Truth CDR': valor_referencia, 'MAE': mae}
        resultados_cdr = resultados_cdr._append(nuevo_resultado_cdr, ignore_index=True)

        total_precision_disco += precision_disco
        total_sensibilidad_disco += sensibilidad_disco
        total_dsc_disco += dsc_disco
        total_similitud_disco += similitud_disco
        total_precision_copa += precision_copa
        total_sensibilidad_copa += sensibilidad_copa
        total_dsc_copa += dsc_copa
        total_similitud_copa += similitud_copa
        total_mae += mae
        print("\n")

    num_images = len(image_names)
    avg_precision_disco = total_precision_disco / num_images
    avg_sensibilidad_disco = total_sensibilidad_disco / num_images
    avg_dsc_disco = total_dsc_disco / num_images
    avg_similitud_disco = total_similitud_disco / num_images
    avg_precision_copa = total_precision_copa / num_images
    avg_sensibilidad_copa = total_sensibilidad_copa / num_images
    avg_dsc_copa = total_dsc_copa / num_images
    avg_similitud_copa = total_similitud_copa / num_images
    avg_mae = total_mae / num_images

    print("\nM Error Absoluto Medio (MAE) en CDR:")
    print(f"MAE: {avg_mae:.2f}")
    print(resultados_cdr)
    print("\n")
    print(resultados_disco)
    print("\n")
    print(resultados_copa)
    print("\n")

    print("Métricas Promedio para el Disco:")
    print(f"Precisión: {avg_precision_disco:.2f}")
    print(f"Sensibilidad: {avg_sensibilidad_disco:.2f}")
    print(f"DSC: {avg_dsc_disco:.2f}")
    print(f"Similitud: {avg_similitud_disco:.2f}")

    print("\nMétricas Promedio para la Copa:")
    print(f"Precisión: {avg_precision_copa:.2f}")
    print(f"Sensibilidad: {avg_sensibilidad_copa:.2f}")
    print(f"DSC: {avg_dsc_copa:.2f}")
    print(f"Similitud: {avg_similitud_copa:.2f}")


image_names = ["g0406", "g0408", "g0414", "g0428", "n0004", "n0005", "n0012", "n0013", "n0014", "n0044", "n0045",
               "n0047", "n0049"]
image_dir = "./refuge_images"
procesar_imagenes(image_names, image_dir)
