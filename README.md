# Algoritmo neuronal de estilos artisticos

## Descripción del Trabajo
El presente trabajo se enfoca en la implementación de un algoritmo que permita la creación de imágenes artísticas de alta calidad perceptiva mediante el uso de redes neuronales profundas.
## Resumen

## Introducción
El ser humano ha logrado crear experiencias visuales a través del arte mediante algoritmos que han resultado un misterio por mucho tiempo. El aprendizaje profundo representa un camino para el entendimiento de este algoritmo, el uso de redes convolucionales.
En este trabajo mostramos que haciendo uso de las redes convolucionales podemos representaer el estilo y el contenido de manera separadas, así podemos generar imágenes mezclando el estilo y el contenido de dos imagenes diferentes.
## Motivo y Descarga de Responsabilidades
El contenido de una imagen y el estilo no se pueden separar completamente, así al queren combinar el contenido de una imagen con el estilo de otra no es posible encontrar una imagen que coincida completamente con ambas restricciones. Por ello para obtener imagenes visualmente atractivas podemos regular el énfasis en el contenido y estilo. Con un fuerte énfasis en el contenido se logra observar claramente la imagen pero no el estido de la otra imagen;y lo mismo ocurre al incrementar el énfasis en el contenido de la otra imagen.

## Metodos
Los resultados de los investigadores Leon A. Gatys, Alexander S. Ecker,  Matthias Bethge fueron realizados en base al uso de una red VGG network, la cualque es una convolutional neural network que ha sido entrenada con aproximandamente 1.2 millones de imagenes del dataset [ImageNet](http://image-net.org/index) por el Visual Geometry Group de la Universidad de Oxford.
El VGG-19 se encuentra disponible en muchas heramientas como caffe, keras, matlab, etc.

En el trabajo se usaron 16 capas convolucionales, 5 capas de agrupamiento de 19 capas VGG.

![](https://image.slidesharecdn.com/adl1103-161027023044/95/applied-deep-learning-1103-convolutional-neural-networks-60-638.jpg?cb=1479405398 "titulo")

Cada capa en la red define a non-linear filter cuya complejidad aumenta con la posición de las capas de red.
Para visualizar que esta codificada en las diferentes capas de jerarquía. Desarrollamos una gradiente de descenso en una white noise image()

Error cuadratíco entre las 2 características
![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima1.png?raw=true)
 - **p :**  imagen orginal
 - **x :**  imagen generada
 - **l :**  capa actual
 - **F<sup>l</sup><sub>ij</sub> :** función de activación del i-esimo filtro en la posición j de la capa l
 - **P<sup>l</sup><sub>ij</sub> :** función de activación de la imagen generada.



Al derivar la función de perdida con respecto a la activación en la capa l.

![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima2.png?raw=true)

- **F<sup>l</sup> :** respresentación de característica de x en la capa l.
- **P<sup>l</sup> :** imagen generada de característica de p en la capa l.
Apartir de la cual podemos calcular la gradiente con respecto  a la imagen utilizando el error estandar back-propagation. Con lo cual podemos cambiar la imagen aleatoria hasta que genere un respuesta en un capa de la CNN como la imagen original p.

Donde F<sup>l</sup> es un matriz en R<sup>N<sub>l</sub>xM<sub>l</sub></sup>
- **N<sub>l</sub>:** Número de filtros distintos
- **M<sub>l</sub>:** tamaño del mapa de características



![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima7.png?raw=true)




Producto interno entre el mapa de carácteristicas vectorizado i y j en la capa l.

![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima3.png?raw=true)



![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima4.png?raw=true)

-  **E<sub>l</sub> :** contribución de la capa a la perdida total.
-  **Lstyle :** Perdida total


Al derivar E<sub>l</sub> con respecto a las activaciones en cada capa.
![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima5.png?raw=true)

Las gradientes de E con respecto con respecto a las activaciones de la capas inferiores pueden ser facilmente calculados usando el error de back-propagation.



Para Generar la mezcla de las imágenes
La función de perdida que minimizamos es:
![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima6.png?raw=true)

## Proceso

![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima10.png?raw=true)

![](https://github.com/Visot/TeoriaAlgoritmica/blob/master/ima/ima11.png?raw=true)

## Conclusiones
- Reemplazar la operación max-pooling por agrupación promedio mejora el flujo de gradiente y se obtiene resultados ligeramente más atractivos.
- Las representaciones de contenido y estilo en la Red Neural Convolucional son separables.
- Las imágenes visualmente más atractivas suelen ser creadas en las capas más altas de la red.
