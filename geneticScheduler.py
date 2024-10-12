import multiprocessing
import random
import time
from racecar3 import Parameters, execute
import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

class WorkerConfig:
	def __init__(self, worker_id,gs):
		self.worker_id = worker_id      # Identificador del worker
		self.params = Parameters()      # Parámetros del algoritmo genético
		self.params.initializeNRandCars(10) 
		self.params.id=len(gs.experiments)
		gs.experiments.append(self.params)             # Resultado que almacenará al terminar


def worker_task(config):
    execute(config.params)
    print(f"Worker {config.worker_id} finished")
    return config


class GeneticScheduler:
		
	def on_task_complete(self,result, task_queue, pool):
		"""Función que se llama cuando un worker completa su tarea."""
		print(f"\nWorker {result.worker_id} completed task")
		# Store the result of the experiment in path
		with open(f"{self.path}/exp{result.params.id}.pkl", "wb") as f:
			del result.params.parameters
			pickle.dump(result.params, f)

		# Asigna un nuevo trabajo tan pronto como el worker termina
		new_task = WorkerConfig(result.worker_id, self)
		print(f"Assigning new task to Worker {result.worker_id}")
		pool.apply_async(worker_task, args=(new_task,), callback=lambda res: self.on_task_complete(res, task_queue, pool))

	def __init__(self,path):
		num_workers = multiprocessing.cpu_count()  # Número de workers según las CPUs

		

		# if not exists path, create it
		if not os.path.exists(path):
			os.makedirs(path)
		self.path = path		

		self.loadExperiments(path)
		#self.study()
		
		"""Scheduler que mantiene a los workers ocupados al 100%."""
		pool = multiprocessing.Pool(processes=num_workers)
		task_queue = multiprocessing.Queue()

		# Inicializamos los workers con sus primeras tareas
		for i in range(num_workers):
			config = WorkerConfig(i,self)  
			pool.apply_async(worker_task, args=(config,), callback=lambda res: self.on_task_complete(res, task_queue, pool))

		# No cerramos ni hacemos join al pool hasta que realmente no se vaya a usar más
		try:
			while True:  # Mantenemos el programa corriendo mientras los workers están activos
				time.sleep(1)  # Puedes ajustar el intervalo de espera
		except KeyboardInterrupt:
			print("Terminating workers...")
			pool.terminate()
		finally:
			pool.join()

	def loadExperiments(self, path):
		self.experiments = []
		# Load experiments from path
		files=os.listdir(path)
		for file in files:
			with open(f"{path}/{file}", "rb") as f: 
					#parse expN.plk
				str_number=re.findall(r'\d+', file)	
				number=int(str_number[0])
				while number>=len(self.experiments):
					self.experiments.append(None)
				self.experiments[number]=pickle.load(f)
		# experiments2=[]
		# for exp in self.experiments:
		# 	if exp!=None:
		# 		experiments2.append(exp)
		# self.experiments=experiments2

	def study(self):
		# Select variable to study
		p = Parameters()    
		for name,values in p.parameters.items():
			if name!="type":
				self.study2(name,values)

	def study2(self, name, values):
		# Inicialización para "regression" y "neural_network"
		count_regression = [0] * len(values)
		sum_regression = [0] * len(values)
		count_nn = [0] * len(values)
		sum_nn = [0] * len(values)
		
		# Recorremos los experimentos y coches
		for exp in self.experiments:
			for car in exp.cars:
					if car["type"] == "pursuit":
						continue  # Saltar "pursuit"
					
					i = values.index(car[name])
					if car["type"] == "regression":
						count_regression[i] += 1
						sum_regression[i] += car["rounds"]
					elif car["type"] == "neural_network":
						count_nn[i] += 1
						sum_nn[i] += car["rounds"]

		# Calcular los promedios para "regression" y "neural_network"
		average_regression = [sum_regression[i] / count_regression[i] if count_regression[i] > 0 else 0 for i in range(len(values))]
		average_nn = [sum_nn[i] / count_nn[i] if count_nn[i] > 0 else 0 for i in range(len(values))]
		
		# Gráfico de barras agrupadas
		bar_width = 0.35
		index = np.arange(len(values))

		plt.bar(index, average_regression, bar_width, label='Regression', color='red')
		plt.bar(index + bar_width, average_nn, bar_width, label='Neural Network', color='yellow')

		# Etiquetas y leyenda
		plt.xlabel(name)
		plt.ylabel("Average rounds")
		plt.title(f"Study of {name}")
		plt.xticks(index + bar_width / 2, values)  # Centrar etiquetas
		plt.legend()

		plt.show()
		time.sleep(1)
		print()
		

if __name__ == "__main__":
    GeneticScheduler("./exp2")
