
function predecir(a, b, c, d){

	inputs = np.empty((10000, 5), float)
	outputs = np.empty((10000, 1), float)

	data1 = data['Content Rating']

	for i in range (10000):
	  inputs[i][0]= data.Category[i] 
	  inputs[i][1]= data.Rating[i]
	  inputs[i][2]= data.Reviews[i]
	  inputs[i][3]= data.Price[i]
	  inputs[i][4]= str(data1[i])
	  outputs[i]=[data.Installs[i]]
	
	capa = tf.keras.layers.Dense(units=5, input_shape=[5]) # El valor indica el numero de entradas
	modelo = tf.keras.Sequential([capa])

	
	modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.5),
    loss='mean_squared_error'
	)


	print("Comenzando entrenamiento...")
	historial = modelo.fit(inputs, outputs, epochs=50, verbose=False)
	print("Modelo entrenado!")


	print("Realizando la predicción...")
	resultado = modelo.predict([[a, b, c, d]])#0.0, 4.1, 159.0, 0.0, 1.0

	result = statistics.mean(resultado[0])
	result = int(result)
	print("")
	print("El resultado es " + str(result) + " -> Número de instalaciones")
	return result
}