# Validación científica final

Generado: 2026-11-18T07:56:59.232793-05:00

## Resultado general

La señal predictiva de `dC/dt` fue evaluada mediante validación agrupada por país, validación temporal, sensibilidad a ventanas futuras y estabilidad de coeficientes.

## Validación temporal

- Pruebas con origen temporal: 5.
- Pruebas positivas: 4 de 5.
- No se utilizó información futura para el entrenamiento.
- La señal fue positiva en cuatro de cinco pruebas, pero falló en una prueba anual con origen 2014.

## Sensibilidad de ventanas

- Pruebas totales: 15.
- Pruebas positivas: 12 de 15.
- Transiciones anuales: 3 de 6 positivas.
- Intervalos de hasta dos años: 9 de 9 positivas.

La señal es más estable cuando se permiten intervalos de hasta dos años. La muestra estrictamente anual presenta mayor variabilidad temporal.

## Estabilidad de coeficientes

- Coincidencia general de signos: 80.0%.
- Variables con estabilidad de signo del 100%: PREVIOUS_C, PREVIOUS_L.
- `PREVIOUS_C` mantuvo signo negativo.
- `PREVIOUS_L` mantuvo signo positivo.
- `PREVIOUS_A` y `PREVIOUS_F` fueron menos estables.

## Interpretación

Los resultados respaldan una señal predictiva pequeña para el cambio en logro educativo adulto. La señal no es uniforme en todos los periodos y no debe interpretarse causalmente.

La evidencia más sólida corresponde a:

1. validación agrupada por país;
2. validación temporal sin información futura;
3. resultados positivos en 9 de 9 pruebas con intervalos de hasta dos años;
4. estabilidad completa de los signos de `PREVIOUS_C` y `PREVIOUS_L`.

## Limitaciones

1. Las transiciones anuales fueron positivas solo en tres de seis pruebas de sensibilidad.
2. El corte anual con origen 2014 produjo un resultado negativo.
3. Los signos de `PREVIOUS_A` y `PREVIOUS_F` no fueron completamente estables.
4. El rendimiento predictivo sigue siendo bajo.
5. Los coeficientes representan asociaciones predictivas, no efectos causales.

## Conclusión

La validación científica final respalda el uso prudente de Ridge para modelar `dC/dt`, especialmente con intervalos de hasta dos años. El resultado debe presentarse como una señal predictiva moderada y parcialmente estable, no como una ley general de dinámica educativa.
