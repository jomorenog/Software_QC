# Software_QC

Este programa está diseñado para realizar las mediciones de MTF, SNR y SDNR; así como para extraer la informacíon del encabezado DICOM, de la imagenes realizadas al objeto de prueba propuesto por el IAEA en su documento "Human Health SeriesNo. 39", para la implementación de un programa de control de calidad remoto y automatizado.

[Ver Documento](https://www-pub.iaea.org/MTCD/Publications/PDF/PUB1936_web.pdf)

## Librerias necesarias:
 pydicom, matplotlib, pandas, numpy, sys, os, cv2, scipy, pillow

## Ejecucio:

Sea *img_name* el nombre del archivo de la imagen DICOM.

Sea *fname* el nombre del atchivo csv donde quiere almacenar los datos.

Ejecute el programa como:

`python3 RW_Dicom.py img_name fname`