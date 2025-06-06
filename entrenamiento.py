
import tensorflow as tf
keras = tf.keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
os.system('cls')

def norm(x):
    #return (x-train_stats['mean'])/(train_stats['std']) 
    return (x-train_stats['min'])/(train_stats['max']-train_stats['min'])



# Obtener lista de GPUs disponibles
gpus = tf.config.list_physical_devices('GPU')

# Mostrar el número de GPUs disponibles
print(f"Número de GPUs disponibles: {len(gpus)}")


df = pd.read_csv("C:\python\kiconex\csvs\data_limpia.csv")
print("Archivo CSV importado con éxito.")
df = df[0:884568] #Elimino el més de abril

df=df.drop_duplicates()



columnas_output = [
    'Temperatura descarga circuito 1 ºC',
    'Temperatura descarga circuito 2 ºC',

    'Temperatura aspiracion circuito 1 ºC',
    'Temperatura aspiracion circuito 2 ºC',

    'Transductor de baja presion circuito 1 (T) ºC',
    'Transductor de baja presion circuito 2 (T) ºC',

    'Transductor de alta presion circuito 1 (T) ºC',
    'Transductor de alta presion circuito 2 (T) ºC',

    'Transductor de baja presion circuito 1 (P) bar',
    'Transductor de baja presion circuito 2 (P) bar',
    'Transductor de alta presion circuito 1 (P) bar',
    'Transductor de alta presion circuito 2 (P) bar',
]
columnas_input = [
    'hora_sin',
    'hora_cos',
    'semana_sin',
    'semana_cos',
    'anio_sin',
    'anio_cos',
    'Modo Unidad',
    'Consigna actual',
    'Velocidad ventilador exterior circuito 1 %',
    'Temperatura exterior ºC',
    'Temperatura retorno general unidad interior ºC',
    'Temperatura impulsion general unidad interior ºC',
    'Apertura valvula C1 %',
    'Apertura valvula C2 %',
    'Demanda compresor inverter 1 %',
    'Demanda compresor inverter 2 %',
]

#  Asegurar que todas las columnas existan
columnas_totales = columnas_input + columnas_output
df = df[columnas_totales].dropna()  # también podés usar fillna si preferís

#  Separar entrenamiento y test
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)

#  Separar inputs y labels
train_labels = train_df[columnas_output].copy()
train_df = train_df[columnas_input].copy()

test_labels = test_df[columnas_output].copy()
test_df = test_df[columnas_input].copy()

#  Estadísticas para normalizar si hace falta
train_stats = train_df.describe().transpose()

# Guardar train_stats en un archivo CSV llamado 'train_stats.csv'
train_df.to_csv('train_df.csv')

print("Datos preparados correctamente:")
print(train_df.shape, train_labels.shape)




norm_train_df=norm(train_df)
norm_test_df=norm(test_df)

n_inp = len(norm_train_df.keys())
n_out = len(columnas_output)

inputs = keras.Input(shape=[n_inp], name='Input')
x = keras.layers.Dense(256,activation='relu')(inputs)
x = keras.layers.Dense(256,activation='relu')(x)
x = keras.layers.Dense(256,activation='relu')(x)
x = keras.layers.Dense(256,activation='relu')(x)
outputs = keras.layers.Dense(n_out)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='model')

model.summary()

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Definición del modelo (asumo que ya lo has definido anteriormente)


model.compile(loss=keras.losses.Huber(),
              optimizer=tf.optimizers.Adam(),  # En la primera fase, el LR por defecto de Adam se utilizará
              metrics=[tf.metrics.MeanAbsoluteError()])

early_stopping = keras.callbacks.EarlyStopping(patience=20)


# Configuración para guardar el mejor modelo durante el entrenamiento
checkpoint = ModelCheckpoint('modelo.h5', 
                             save_best_only=True,  # Guarda el mejor modelo según val_loss
                             save_weights_only=False,  # Guardar todo el modelo (no solo los pesos)
                             verbose=1)

history = model.fit(
    norm_train_df,
    train_labels,
    batch_size=2**10,
    epochs=5000,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],  # Incluyendo el callback para guardar el modelo
    verbose=1
)





import matplotlib.pyplot as plt

# Generación de la figura
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida. validación')
plt.xlabel('Época', fontsize=20)
plt.ylabel('Función de Pérdida.', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
# Guardar la figura en un archivo .png
#plt.savefig('grafico_perdida.png', dpi=300)  # Guardamos el gráfico con una resolución de 300 dpi
plt.show()



loss= model.evaluate(norm_test_df, test_labels) 




