# El siguiente código tiene el propósito de generar
# los histogramas de la variable_1 y la variable_2
# y generar las líneas de mejor ajuste para estos


# http://127.0.0.1:8000/ este va a ser el link del trabajo
# donde está el documento que corresponde
# al avance, de igual manera se va subir una archivo con el link.
# Gabriel González Rivera - B93432
# Dilana Rodríguez Jiménez - C06660
# Sebastián Bonilla Vega - C01263

# Se importan las liberías necesarias
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from scipy.stats import skew, kurtosis
import numpy as np


#  se conecta a la base de datos usando sqlite3.connect
conn = sqlite3.connect('proyecto.db')

# Consulta para extraer los datos de las dos variables
query = "SELECT variable_1, variable_2 FROM test_data"
data = pd.read_sql_query(query, conn)

# Cerrar la conexión con la base de datos
conn.close()

# Mostrar las primeras filas de los datos
# para comprobar que estos se extrajeron
print(data.head())

# Crear histogramas de variable_1 y variable_2

# Primero se especifica el tamaño de la figura
plt.figure(figsize=(12, 6))


# Histograma original de variable_1
# Se genera el primer gráfico de dos de la imagen en la izquierda
plt.subplot(1, 2, 1)

# Se genera el gráfico con 20 barras en color azul y con transparencia 0.7 de
# la variable_1
plt.hist(data['variable_1'], bins=20, color='blue', alpha=0.7)

# Se pone el título de la gráfica y los títulos de los ejes
# usando title y label
plt.title('Histograma de variable 1 (Original)')
plt.xlabel('variable 1')
plt.ylabel('Frecuencia')


# Histograma original de variable_2
# Para este histograma se sigue un procedimiento similar al anterior
plt.subplot(1, 2, 2)
plt.hist(data['variable_2'], bins=20, color='green', alpha=0.7)
plt.title('Histograma de variable 2 (Original)')
plt.xlabel('variable 2')
plt.ylabel('Frecuencia')

# Se guarda la figura de los histogramas originales como una imagen .png
plt.savefig('histogramas_originales.png')


# Se crea una nueva figura para los histogramas con las líneas de mejor ajuste
plt.figure(figsize=(12, 6))

# Histograma con línea de mejor ajuste para variable_1
plt.subplot(1, 2, 1)

# Se genera un histograma como el anterior pero se guarda
# el conteo de las barras en count1 y el conteo de sus
# bordes en bins1, ignored1 se usa para guardar los
# valores retornados pero no usados de los histogramas
count1, bins1, ignored1 = plt.hist(data['variable_1'], bins=20,
                                   color='blue', alpha=0.7, density=False)

# Se adaptan los datos a una distribución normal y se guarda
# la media en mu_1 y la desviación estándar en std_1
mu_1, std_1 = norm.fit(data['variable_1'])

# Se calcula la línea de mejor ajuste con respecto a
# los datos obtenidos anteriormente y se ajusta para
# que calce con el histograma con la multiplicación
# posterior.
best_fit_line1 = (norm.pdf(bins1, mu_1, std_1) *
                  count1.sum() * np.diff(bins1)[0])

# Se dibuja la línea de ajuste en negro con un grosor de dos
plt.plot(bins1, best_fit_line1, 'k', linewidth=2)
plt.title('Histograma de variable 1 con ajuste gaussiano')
plt.xlabel('variable 1')
plt.ylabel('Frecuencia')

# Histograma de variable_2 con curva de ajuste

# Para la variable_2 se sigue un procedimiento similar al de
# el histograma pasado, solo que en vez de una distribución
# normal se usa una exponencial.
plt.subplot(1, 2, 2)
count2, bins2, ignored2 = (plt.hist(data['variable_2'], bins=20,
                                    color='green', alpha=0.7, density=False))
loc_2, scale_2 = expon.fit(data['variable_2'], floc=0)

# Se fija loc a 0 para mejorar el ajuste
best_fit_line2 = (expon.pdf(bins2, loc_2, scale_2)
                  * count2.sum() * np.diff(bins2)[0])
plt.plot(bins2, best_fit_line2, 'k', linewidth=2)
plt.title('Histograma de variable 2 con ajuste exponencial')
plt.xlabel('variable 2')
plt.ylabel('Frecuencia')
plt.savefig('histogramas_mejor_ajuste.png')


# Graficar histograma y la curva ajustada
plt.figure(figsize=(6, 4))


# Cálculo de momentos de la variable_1
# se calcula cada momento necesario mediante
# las funciones correspondientes
mean_1 = data['variable_1'].mean()
var_1 = data['variable_1'].var()
std_1 = data['variable_1'].std()
skew_1 = skew(data['variable_1'])
kurt_1 = kurtosis(data['variable_1'])

# Cálculo de momentos de la variable_2
# se calculan los momentos de manera similar
mean_2 = data['variable_2'].mean()
var_2 = data['variable_2'].var()
std_2 = data['variable_2'].std()
skew_2 = skew(data['variable_2'])
kurt_2 = kurtosis(data['variable_2'])

# Ahora se muestran los resultados de ambas
# variables

# Mostrar resultados variable_1
print('Momentos de variable_1:')
print(f'Promedio: {mean_1}')
print(f'Varianza: {var_1}')
print(f'Desviación estándar: {std_1}')
print(f'Inclinación: {skew_1}')
print(f'Kurtosis: {kurt_1}')

# Mostrar resultados variable_2
print('Momentos de variable_2:')
print(f'Promedio: {mean_2}')
print(f'Varianza: {var_2}')
print(f'Desviación estándar: {std_2}')
print(f'Inclinación: {skew_2}')
print(f'Kurtosis: {kurt_2}')
