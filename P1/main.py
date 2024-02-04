
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color

count_image = 0
count_histogram = 0


# Función para obtener la imagen de entrada
def get_image_path(path):
    image = io.imread(path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]  # Eliminamos el canal alfa
        if image.shape[2] == 3:
            image = color.rgb2gray(image)
    elif image.ndim == 2:
        image = normalize(image, np.min(image), np.max(image))
    return image

# Función de normalización
def normalize(image, min_val, max_val):  # Cogemos el el mínimo y máximo de la imagen y ajusta el rango
    image = (image - min_val) / (max_val - min_val)
    return image

# Plot y guardado de imágenes
def plotImage(image, outimage):
    global count_image
    plt.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    plt.show()
    plt.imshow(outimage, cmap='gray', vmin=0.0, vmax=1.0)
    plt.show()
    count_image += 1
    io.imsave(f"./outputs/output_image_{count_image}.png", (outimage * 255).astype(np.uint8))

# Plot histograma
def histogram(image, bins=256):
    global count_histogram
    plt.hist(image.ravel(), bins=bins, range=(0.0, 1.0))
    count_histogram += 1
    plt.savefig(f"./outputs/output_histogram_{count_histogram}.png")
    plt.show()

# Plot histograma acumulado
def histacumulado(image, bins=256):
    out_hist, _ = np.histogram(image.ravel(), bins=bins, range=(0.0, 1.0))
    out_cdf = out_hist.cumsum()
    out_cdf_normalized = out_cdf / float(out_cdf.max())
    plt.figure(figsize=(8, 6))
    plt.plot(range(bins), out_cdf_normalized)  # Plot del histograma acumulado de la imagen
    plt.title('Histograma Acumulado')
    plt.grid()
    plt.savefig(f"./outputs/output_histacumulado_{count_histogram}.png")
    plt.show()

# Plot kernel Gaussiano Unidimensional en función de σ
def plotk(k, sigma):
    plt.plot(k)
    plt.title(f'Gauss Kernel 1D (σ = {sigma})')
    plt.xlabel('Posición en el kernel')
    plt.ylabel('Valor del kernel')
    plt.grid()
    plt.savefig(f"./outputs/output_gaussKernel1D.png")
    plt.show()

# Función 1: adjustIntensity

def adjustIntensity(inImage, inRange=[], outRange=[0.0, 1.0]):  # Transformación lineal de lo valores de  intensidad para
                                                                # que se ajusten al nuevo rango, comprensión o estiramiento lineal de histograma
    if not inRange:
        imin = float(np.min(inImage))
        imax = float(np.max(inImage))
    else:
        imin, imax = [float(val) for val in inRange]

    omin, omax = [float(val) for val in outRange]

    outImage = omin + ((omax - omin) * (inImage - imin) / (imax - imin))
    return outImage

"""
inImage = get_image_path("./fotos_prueba/gato.png")
outImage = adjustIntensity(inImage,[],[0.4,0.8])
plotImage(inImage, outImage)
histogram(inImage)
histogram(outImage)
"""

# Función 2: equalizeIntensity

def equalizeIntensity(inImage, nBins=256):  # Ecualización de histograma
    hist, bins = np.histogram(inImage.ravel(), bins=nBins)
    cdf = hist.cumsum()  # Se calcula el cdf, func de distribución acumulativa y luego lo normalizamos
    cdf_normalized = cdf / float(cdf.max())
    outImage = np.interp(inImage, bins[:-1],
                         cdf_normalized)  # Después hacemos una interp. lineal para cada valor de intensidad de inImage teniendo en cuenta el cdf
    return outImage

"""
nBins = 230
inImage = get_image_path("./fotos_prueba/gato.png")
outImage = equalizeIntensity(inImage,nBins)
plotImage(inImage, outImage)
histogram(inImage)
histogram(outImage, nBins)
histacumulado(outImage, nBins)
"""

# Función 3: filterImage
def filterImage(inImage, kernel):  # Filtrado espacial mediante convolución
    global count_image

    M, N = inImage.shape
    k_M, k_N = kernel.shape
    M_relleno = k_M // 2  # Cantidad de relleno
    N_relleno = k_N // 2
    inImage_pad = np.pad(inImage, ((M_relleno, M_relleno), (N_relleno, N_relleno)), mode='reflect')  # Aplicamos relleno reflectante para  manejar bordes correctamente
    outImage = np.zeros_like(inImage, dtype=np.float64)

    for i in range(M):
        for j in range(N):
            result = 0.0
            for m in range(k_M):
                for n in range(k_N):
                    result += inImage_pad[i + m, j + n] * kernel[m, n]  # para cada pixel de la imagen, result acumula la suma ponderada del producto del elemento del kernel en esa pos
                                                                        # y el valor de la imagen en la region cubierta por el kernel, lo acumula mientras recorremos el kernel
            outImage[i, j] = result  # aplicamos el resultado al pixel después de iterar sobre cada pos del kernel

    return outImage


"""
# en este kernel habría que normalizar porque se sale de rango
k1 = np.ones((10, 10), dtype=np.float32) / 9.0

# en este no habría que normalizar porque no se sale de rango
k2 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]], dtype=np.float32) / 9.0

inImage = get_image_path("./fotos_prueba/gato.png")
outImage = filterImage(inImage, k1)
outImage = normalize(outImage,np.min(outImage),np.max(outImage))
plotImage(inImage, outImage)
"""

# Función 4: gaussKernel1D
def gaussKernel1D(sigma):  # kernel Gaussiano unidimensional
    N = int(2 * np.ceil(3 * sigma) + 1)  # Tamaño del kernel en función de σ
    center = N // 2  # Centro del kernel para garantizar el centro de la Gaussiana
    kernel = np.zeros((N, 1))

    for x in range(N):
        exponente = -((x - center) ** 2) / (2 * sigma ** 2)  # calculamos el exp de la gaussiana en esa pos
        kernel[x] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(exponente)  # valor de la función Gaussiana en esa pos

    return kernel

"""
sigma = 20
kernel = gaussKernel1D(sigma)
plotk(kernel,sigma)
"""

# Función 5: gaussianFilter

def gaussianFilter(inImage, sigma):  # Suavizado Gaussiano bidimensional, podemos hacer que el kernel sea de Nx1 y 1xN para poder convolucionar
    k = gaussKernel1D(sigma)
    img_vert = filterImage(inImage, k)  # Aplicar el kernel 1D verticalmente NX1
    k_hor = k.T  # Transponer el kernel para el filtrado horizontal
    outImage = filterImage(img_vert, k_hor)  # Aplicar el kernel 1D horizontalmente 1XN
    return outImage

"""
sigma = 1.5
inImage = get_image_path("./fotos_prueba/lenna_ruido.png")
outImage = gaussianFilter(inImage, sigma)
plotImage(inImage, outImage)
"""

# Función 6: medianFilter

def medianFilter(inImage, filterSize):  # Filtro de medianas bidimensional, el tamaño de la ventana será de NxN
    M, N = inImage.shape
    relleno = filterSize // 2  # relleno que se usará
    outImage = np.zeros_like(inImage)
    inImage_pad = np.pad(inImage, ((relleno, relleno), (relleno, relleno)), mode='reflect')  # Aplicamos relleno reflectante para manejar bordes

    for i in range(M):
        for j in range(N):
            ventana = inImage_pad[i:i + filterSize, j:j + filterSize]  # Ventana de píxeles centrada en el píxel actual
            val = np.median(ventana)  # Mediana de la ventana
            outImage[i, j] = val  # Valor de la mediana a la posición central de la ventana en la imagen de salida

    return outImage

"""
filterSize = 5
inImage = get_image_path("./fotos_prueba/lenna_ruido.png")
outImage = medianFilter(inImage, filterSize)
plotImage(inImage, outImage)
"""

def operacion_morfologica(inImage, SE, operacion, center=[]):
    if not center:
        center = (SE.shape[0] // 2, SE.shape[1] // 2)

    inImage = np.round(inImage)  # convertimos la im a binario
    SE_M, SE_N = SE.shape
    M, N = inImage.shape
    outImage = np.zeros_like(inImage)

    relleno_M = SE_M // 2
    relleno_N = SE_N // 2

    inImage_pad = np.pad(inImage, ((relleno_M, relleno_M), (relleno_N, relleno_N)), mode='constant', constant_values=0)  # rellenamos con 0 para manejar bordes

    if center[0] < 0 or center[0] >= SE_M or center[1] < 0 or center[1] >= SE_N:
        raise ValueError("El centro no está dentro de los valores permitidos del elemento estructurante")

    for i in range(M):
        for j in range(N):
            if operacion == "erode":
                min_val = 1  # max val que puede tomar
                for m in range(SE_M):
                    for n in range(SE_N):
                        if SE[m, n] == 1:  # si esa pos del elem estructurante está activa
                            val = inImage_pad[i - center[0] + m + relleno_M, j - center[
                                1] + n + relleno_N]  # valor de la im de entrada teniendo en cuenta la pos en el SE
                            min_val = min(min_val, val)  # cogemos el minimo, si es 0 erosionamos
                outImage[i, j] = min_val

            elif operacion == "dilate":
                max_val = 0  # min val que puede tomar
                for m in range(SE_M):
                    for n in range(SE_N):
                        if SE[m, n] == 1:
                            val = inImage_pad[i - center[0] + m + relleno_M, j - center[1] + n + relleno_N]
                            max_val = max(max_val, val)  # cogemos el max, si es 1 dilatamos
                outImage[i, j] = max_val

    return outImage

# Función 7: erode
def erode(inImage, SE, center=[]):
    return operacion_morfologica(inImage, SE, "erode", center)


"""
inImage = get_image_path("./fotos_prueba/animo.jpeg")
SE = np.array([[1,1,1],
               [1,1,1],
               [1,1,1]])

outImage = erode(inImage, SE)
plotImage(inImage, outImage)
"""

# Función 8: dilate
def dilate(inImage, SE, center=[]):
    return operacion_morfologica(inImage, SE, "dilate", center)

"""
inImage = get_image_path("./fotos_prueba/animo.jpeg")
SE = np.array([[1,1,1],
               [1,1,1],
               [1,1,1]])

outImage = dilate(inImage, SE)
plotImage(inImage, outImage)
"""

# Función 9: opening
def opening(inImage, SE, center=[]):
    eroded = erode(inImage, SE, center)  # eliminamos pequeños detalles y abrimos separaciones
    opened = dilate(eroded, SE, center)  # restauramos tamaño pero con los cambios anteriores aplicados
    return opened


"""
inImage = get_image_path("./fotos_prueba/animo.jpeg")
SE = np.array([[1,1,1],
               [1,1,1],
               [1,1,1]])
outImage = opening(inImage,SE)
plotImage(inImage,outImage)
"""

# Función 10: closing

def closing(inImage, SE, center=[]):
    dilated = dilate(inImage, SE, center)  # conectamos regiones al expandir
    closed = erode(dilated, SE, center)  # restauramos tamaño pero con los cambios anteriores aplicados
    return closed


"""
inImage = get_image_path("./fotos_prueba/animo.jpeg")
SE = np.array([[1,1,1],
               [1,1,1],
               [1,1,1]])
outImage = closing(inImage, SE)
plotImage(inImage,outImage)
"""

# Función 11: hit or miss

def hit_or_miss(inImage, objSE, bgSE, center=[]):  # buscamos pixeles que cumplan los patrones de objSE y bgSE
    if not center:
        center = (objSE.shape[0] // 2, objSE.shape[1] // 2)

    inImage = np.round(inImage)
    if np.any(np.logical_and(objSE, bgSE)):  # no pueden coincidir los 1s en las mismas pos de los SE
        raise ValueError("Error: elementos estructurantes incoherentes")

    eroded_obj = erode(inImage, objSE, center)
    eroded_bg = erode(1 - inImage, bgSE, center)  # invertimos la imagen para buscar el fondo
    outImage = np.logical_and(eroded_obj, eroded_bg)  # im con los pixeles que cumplen objSE y bgSE

    return outImage


"""
objSE = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0]])

bgSE = np.array([[0, 0, 1],
                 [0, 1, 0],
                 [0, 0, 0]])

inImage = get_image_path("./fotos_prueba/image.png")
outImage = hit_or_miss(inImage, objSE, bgSE)
plotImage(inImage, outImage)
"""

# Función 12: gradientImage
def gradientImage(inImage, operator):  # Obtiene las componentes Gx y Gy del gradiente de una imagen
    if operator == 'Roberts':
        gx = np.array([[-1, 0], [0, 1]])
        gy = np.array([[0, -1], [1, 0]])
    elif operator == 'CentralDiff':
        gx = np.array([[-1, 0, 1]])
        gy = gx.T
    elif operator == 'Prewitt':
        gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif operator == 'Sobel':
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    else:
        raise ValueError("Operador no válido, debe ser 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'.")

    gx = filterImage(inImage, gx)  # convolucionamos para obtener las componentes
    gy = filterImage(inImage, gy)

    return [gx, gy]


"""
inImage = get_image_path("./fotos_prueba/punto1.jpg")
[gx, gy] = gradientImage(inImage, "Sobel")
plt.imshow(inImage, cmap='gray',vmin=0.0,vmax=1.0)
plt.show()
plt.imshow(gx, cmap='gray')
plt.show()
plt.imshow(gy, cmap='gray')
plt.show()
"""

# Función 13: LoG

def LoG(inImage, sigma):  # Filtro Laplaciano de Gaussiano en función de σ

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # definimos el kernel laplaciano
    imagen_suavizada = gaussianFilter(inImage, sigma)  # Aplicamos filtro Gaussiano
    outImage = filterImage(imagen_suavizada, kernel)  # Aplicamos el filtro Laplaciano de LoG a la imagen suavizada
    outImage = cruces_por_cero(outImage,
                               0.01)  # Buscamos cruces por cero (donde la imagen cambia, lo que indica bordes)
    return outImage

def cruces_por_cero(image, t):
    cruces = np.zeros(image.shape, dtype=float)
    M, N = image.shape
    n = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
    for x in range(1, M - 1):
        for y in range(1, N - 1):
            posicion = image[x, y]
            if posicion < -t:  # miramos si el pixel en el que estamos es menor que el umbral neg
                if any(image[x + x_i, y + y_j] > t for x_i, y_j in
                       n):  # si es así, miramos si alguno de los 8 vecinos es mayor que el umbral positivo
                    cruces[x, y] = 1.0
    return cruces


"""
# estoy utilizando un t 0.01 para cruces por cero
inImage = get_image_path("./fotos_prueba/lenna.jpeg")
outImage = LoG(inImage,1.5)
plotImage(inImage, outImage)
"""

# Función 14: edgeCanny
def snm(M, G): # M es la magnitud gradiente y G es la direccion del gradiente
    SNM = np.zeros(M.shape)
    direcciones = []

    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            angulo = G[i, j] # cogemos el angulo de la direcc del gradiente en el pixel actual
            n1 = M[i, j]
            n2 = M[i, j]

            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                Dk = (0, 1)  # Vertical
                n1 = M[i, j - 1] # seleccionamos vecinos en la direcc normal al gradiente , dependiendo de la direcc del gradiente
                n2 = M[i, j + 1]
            elif 22.5 <= angulo < 67.5:
                Dk = (-1, 1)  # Diagonal
                n1 = M[i + 1, j + 1]
                n2 = M[i - 1, j - 1]
            elif 67.5 <= angulo < 112.5:
                Dk = (-1, 0)  # Horizontal
                n1 = M[i - 1, j]
                n2 = M[i + 1, j]
            elif 112.5 <= angulo < 157.5:
                Dk = (-1, -1)  # Diagonal
                n1 = M[i + 1, j - 1]
                n2 = M[i - 1, j + 1]

            if M[i, j] < n1 or M[i, j] < n2: # comparamos el pixel con los vecinos de la direcc normal al gradiente, si es menor
                                            # que alguno de ellos, establecemos como 0 haciendo asi supresion de no maximos
                SNM[i, j] = 0
            else:
                SNM[i, j] = M[i, j] # de lo contrario conservamos

            direcciones.append(Dk)  # guardamos el Dk del pixel actual en direcciones

    return SNM, direcciones # devolvemos la matriz con los no maximos eliminados y las direcciones


def hysteresis(image, low, high, direcciones):
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.float64)
    fuerte = 1.0
    fuerte_i, fuerte_j = np.where(image > high) # buscamos pixeles donde sean mayores que high
    result[fuerte_i, fuerte_j] = fuerte # los asignamos como fuertes

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] > high: # los pixeles que sean mayor que el umbral alto
                # usamos la dirección almacenada en direcciones
                index = i * (N - 2) + j - 1
                if 0 <= index < len(direcciones):
                    Dk = direcciones[index]
                    n1 = (-Dk[1], Dk[0]) #calculamos los vecinos en la direcc perpendicular a la normal
                    n2 = (Dk[1], -Dk[0])

                # Verificamos si al menos uno de los vecinos es fuerte
                if result[i + n1[0], j + n1[1]] == fuerte or result[i + n2[0], j + n2[1]] == fuerte:

                    result[i, j] = fuerte # si es asi, asignamos el pixel actual como fuerte

                    # Hacemos un seguimiento de los píxeles conectados usando n1
                    current_i, current_j = i + n1[0], j + n1[1]
                    while 0 <= current_i < M and 0 <= current_j < N and image[current_i, current_j] > low: #miramos que el actual sea mayor que low
                        result[current_i, current_j] = fuerte
                        current_i += n1[0]
                        current_j += n1[1]

                    # Hacemos un seguimiento de los píxeles conectados usando n2
                    current_i, current_j = i + n2[0], j + n2[1]
                    while 0 <= current_i < M and 0 <= current_j < N and image[current_i, current_j] > low:
                        result[current_i, current_j] = fuerte
                        current_i += n2[0]
                        current_j += n2[1]
    return result


def edgeCanny(inImage, sigma, tlow, thigh):
    imagen_suavizada = gaussianFilter(inImage, sigma) #aplicamos primero filtro Gaussiano a la imagen
    [gx, gy] = gradientImage(imagen_suavizada, operator='Sobel') #obtenemos gx y gy para luego obtener la magnitud del gradiente
    G = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    angulo = np.degrees(theta) #convertimos a grados y acotamos a 180
    angulo = np.where(angulo < 0, angulo + 180, angulo)
    imagen_snm, val_dk = snm(G, angulo) # obtenemos la imagen con la supresion de no maximos y las direcciones y los usamos en la histeresis
    outImage = hysteresis(imagen_snm, tlow, thigh, val_dk)
    return outImage

"""
inImage = get_image_path("./fotos_prueba/matricula.png")
outImage = edgeCanny(inImage, 0.5, 0.1, 0.8)
plotImage(inImage, outImage)
"""

# Función 15: cornerSusan

def cornerSusan(inImage, r, t): # r radio de la máscara circular, define tamaño de la vecindad local que se mirará en cada pixel
                                # t umbral de diferencia de intensidad respecto al nucleo de la máscara, dif maxima permitida entre
                                # intensidad del pixel central y los pixeles en la vecindad
    M, N = inImage.shape
    outCorners = np.zeros_like(inImage, dtype=np.float32) # detector de esquinas
    usanArea = np.zeros_like(inImage, dtype=np.float32)   # area usan

    for i in range(r, M - r): #excepto borde de tamaño r
        for j in range(r, N - r):

            if i - r >= 0 and i + r + 1 <= M and j - r >= 0 and j + r + 1 <= N:  # Aseguramos que la region de radio r alrededor del pixel esté dentro de los límites de la imagen
                reg = inImage[i - r:i + r + 1, j - r:j + r + 1]   # extraemos la region  local alrededor del pixel: (2*r + 1) x (2*r + 1)
                y, x = np.mgrid[-r:r + 1, -r:r + 1]    # Creamos una rejilla de coordenadas y e x correspondientes a la reg local
                mascara = x ** 2 + y ** 2 <= r ** 2    # Calculamos la máscara circular que incluye los pixeles dentro del r dado
                usan = np.sum(mascara * (np.abs(reg - inImage[i, j]) <= t)) # Calculamos el área usan contando los píxeles dentro del área circular que cumplen con la condición de diferencia de intensidad
                g = 3 / 4
                outCorners[i, j] = max(0, g - (usan / np.sum(mascara))) #maximo entre 0 y la dif del umbral geometrico con el area normalizada

                usanArea[i, j] = usan / np.sum(mascara) # actualizamos el area normalizando por el total de pixeles de la mascara circular

    return outCorners, usanArea

"""
inImage = get_image_path("./fotos_prueba/cuadrado1.jpeg")
outCorners, usanArea = cornerSusan(inImage, 7, 0.5)
outCorners = normalize(outCorners, np.min(outCorners), np.max(outCorners))
usanArea = normalize(usanArea, np.min(usanArea), np.max(usanArea))
plt.imshow(inImage, cmap='gray', vmin=0.0, vmax=1.0)
plt.show()
plotImage(outCorners, usanArea)
"""
