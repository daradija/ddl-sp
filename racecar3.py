from drnumba import *


# Derives from version 2, include punishment for hitting the walls
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Deshabilita la GPU

import random
import math
import numpy as np
import time

drnumba=DrNumba("kernel.py")

# Set the dimensions of the window
screen_width = 800 * 2
screen_height = 500 * 2

grade= 1
depth = 1


# Define colors
black = (0, 0, 0)

def sign(x):
	if x>0:
		return 1
	if x<0:
		return -1
	return 0

class Pursuit:
	def __init__(self):
		pass

	def usable(self):
		return False
	
	def add(self, xs, vel, heuristicDistance, leftb, rightb, upb):
		pass

class Regressor:
	def __init__(self):
		# Maximum number of training samples
		#self.max_samples = 10000
		
		# Minimum number of training samples for pruning
		# self.min_samples = 1000

		# Lists to store training data
		self.xs = []
		self.ys = []

		# Initial target position
		self.target_position = None

		# Regression coefficients (weights)
		self.B = None

	def usable(self):
		return self.B is not None

	def similar(self, a, b):
		return min(a, b) /max(a, b)
 
	def predict(self, xs,param):
		"""
		Predict the output based on input features using the regression model.
		
		Parameters:
			xs (list): Input features
		
		Returns:
			list: Predicted outputs
		"""
		keys = [[0, 0, 1],
				[0, 1, 1],
				# [1, 0, 0],
				# [0, 1, 0],
				[1, 0, 1]]
		base = [(None, xs)]
		best_heuristic = -math.inf
		r = None
		for prof in range(depth):
			base2 = []
			for k, b in base:
				if prof>0:
					keys=[k]
				for key in keys:
					xs2 = np.array( upgrade(grade,b+key)) 
					ys = np.dot(xs2.T, self.B)
					# punish=ys[:7].sum()/7
					# if punish<1:
					# 	punish=1
					# punish2=ys[8] #+ys[8]/10
					if prof == depth - 1:
						#heuristic = self.similar(ys[0], ys[6]) * self.similar(ys[1], ys[5]) * self.similar(ys[2], ys[4]) * ys[3]* sign(ys[8])* ys[7]
						
						num_arms = param["lidar"]
						heuristic = 1 
						for i in range(num_arms // 2):
							heuristic *= self.similar(ys[i], ys[num_arms - 1 - i])
						# if is odd
						if num_arms % 2 == 1:
							heuristic *= ys[num_arms // 2] 
						heuristic *=  sign(ys[num_arms+1])
						if param["velocity"]==1:
							heuristic *= ys[num_arms]

						#heuristic = punish2
						if heuristic > best_heuristic:
							best_heuristic = heuristic
							if prof==0:
								r = key
							else:
								r = k
					else:
						if prof == 0:
							base2.append((key, ys.tolist()))
						else:
							base2.append((k, ys.tolist()))
			#keys=[(0,0,1)]
			base = base2
		#print(best_heuristic)
		return (True if r2 == 1 else False for r2 in r)

	def add(self, xs, vel, heuristicDistance, leftb, rightb, upb):
		"""
		Add a new training sample to the regression model.
		
		Parameters:
			xs (list): Input features
			vel (float): Velocity
			leftb (bool): Left button pressed
			rightb (bool): Right button pressed
			upb (bool): Up button pressed
		
		Returns:
			None
		"""
		if self.B is not None:
			return
		if not self.target_position:
			self.target_position = xs    
			self.target_vel=vel
			self.target_heuristicDistance=heuristicDistance
			return 
		left = 1 if leftb else 0
		right = 1 if rightb else 0
		up = 1 if upb else 0
		self.xs.append(upgrade(grade,self.target_position + [self.target_vel,self.target_heuristicDistance , left, right, up]))
		self.ys.append(xs + [vel, heuristicDistance])
		if len(self.xs) == self.initial_train:
			X = np.array(self.xs)
			Y = np.array(self.ys)
			self.B = np.linalg.lstsq(X, Y, rcond=None)[0]
			
			# xs2 = []
			# ys2 = []
			# for i in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
			# 	num = 0
			# 	for j in range(len(self.xs) - 1, -1, -1):
			# 		xs3 = self.xs[j]
			# 		tu = tuple(xs3[-3:])
			# 		if tu == i:
			# 			num += 1
			# 			xs2.append(xs3)
			# 			ys3 = self.ys[j]
			# 			ys2.append(ys3)
			# 		if self.min_samples / 8 <= num:
			# 			break
			# self.xs = xs2
			# self.ys = ys2

		self.target_position = xs
		self.target_vel=vel
		self.target_heuristicDistance=heuristicDistance

class NumbaNN:
	def __init__(self,model):
		self.dr=drnumba.dr(self)

		weights = model.get_weights()

		maxWidth=model.layers[0].input.shape[1]+1
		maxHeight=0
		for layer in model.layers:
			maxHeight+=1
			if layer.input.shape[1]>maxWidth:
				maxWidth=layer.input.shape[1]+1

		self.h=maxHeight
		self.w=maxHeight
		self.weights = np.zeros((maxHeight,maxWidth,maxWidth),dtype=np.float16)

		self.data=np.zeros(maxWidth,dtype=np.float16)

	 	#par entrada salida
		#impar salida
		for i,h in enumerate(weights):
			for j,w in enumerate(h):
				if i%2==0:
					for k,v in enumerate(w):
						self.weights[i//2][k][j]=v
				else:
					self.weights[i//2][j][maxWidth-1]=w

		#self.dr.data("h","w","w","weights")
		#self.dr.data("w","data",param=["predict"])

		# identifica si hay que hacer activación
		self.activation=np.zeros(maxHeight,dtype=np.int8)

		code=["linear","sigmoid","relu","softmax","tanh"]

		for i,layer in enumerate(model.layers):
			# print(f"Capa: {layer.name}")
			# print(f"Tipo de capa: {layer.__class__.__name__}")
			# print(f"Entradas: {layer.input.shape}")
			self.lastWidh=layer.output.shape[1]
			# print(f"Función de activación: {layer.activation if hasattr(layer, 'activation') else 'No tiene'}")
			# # get function name:
			# print(layer.activation.__name__)
			self.activation[i]=code.index(layer.activation.__name__)
			# print("\n")
		#self.dr.data("h","activation")
		#self.dr.function("predict2","w")


	def predict(self, xs): # version numpy
		# copy xs to self.data
		data=np.zeros(self.data.shape,dtype=np.float16)
		data2=np.zeros(self.data.shape,dtype=np.float16)
		data[:len(xs)]=xs

		

		sigmoide=np.vectorize( lambda x: (1/(1+np.exp(-x))) if -11<x else 0 if x<11 else 1  )

		# vectorial product
		for i in range(self.weights.shape[0]):
			data[-1]=1
			for idx in range(self.weights.shape[1]):
				# scalar product
				c=np.dot(self.weights[i][idx],data)
				
				data2[idx]=c
			if self.activation[i]==1: # sigmoid
				data2=sigmoide(data2)
			data[:]=data2

		return data[:self.lastWidh]

	def predict2(self):
		idx=cuda.grid(1)
		if idx>=self.weights.shape[1]:
			return
		
		for i in range(self.weights.shape[0]):
			c=np.float16(idx==self.weights.shape[1]-1)
			for j in range(self.weights.shape[2]):
				c+=self.weights[i][idx][j]*self.data[j]

			if self.activation[i]==1: # sigmoid
				c=1/(1+math.exp(-c))
			
			cuda.syncthreads()
			self.data[j]=c
			cuda.syncthreads()

class NeuralNetwork:
	def __init__(self):
		self.xs = []
		self.initial_train = 2000
		self.old = None

	def predict2(self, xs):
		model= self.model
		X_test = np.array([xs])
		predicciones = model.predict(X_test, verbose=0)
		r= predicciones.tolist()[0]
		return r

	def similar(self, a, b):
		if a<0:
			a=0
		if b<0:
			b=0
		if a==0 and b==0:
			return 1
		return min(a, b) /max(a, b)

	def predictV2(self, xs):
		keys = [[0, 0, 1],
			[0, 1, 1],
			# [1, 0, 0],
			# [0, 1, 0],
			[1, 0, 1]]
		best_heuristic = -math.inf
		r = None
		for key in keys:	
			#xs2=self.xs[-10:]
			#xs2.append(xs+key)
			#prediciones=self.model.predict(np.array(xs2), verbose=0)
			start=time.time()
			prediciones=self.model.predict(np.array([xs+key]), verbose=0)
			print(time.time()-start)
			heuristic = prediciones[-1,8]
			if heuristic > best_heuristic:
				best_heuristic = heuristic
				r = key
		return (True if r2 == 1 else False for r2 in r)

	def predict(self, xs,param):
		"""
		Predict the output based on input features using the regression model.
		
		Parameters:
			xs (list): Input features
		
		Returns:
			list: Predicted outputs
		"""
		keys = [[0, 0, 1],
				[0, 1, 1],
				# [1, 0, 0],
				# [0, 1, 0],
				[1, 0, 1]]
		base = [(None, xs)]
		best_heuristic = -math.inf
		r = None
		for prof in range(depth):
			# if prof>0:
			# 	keys=[k]
			base2 = []
			xs2=[]
			for k, b in base:
				
				for key in keys:
					xs2.append(b+key)
			# start=time.time()
			#ys2=self.model.predict(np.array(xs2), verbose=0)	
			# print(time.time()-start)		

			if not hasattr(self, "nnn"):
				self.nnn=NumbaNN(self.model)
			nnn=self.nnn
			#start=time.time()

			#ys2=[nnn.predict(np.array(xs2[0])),nnn.predict(np.array(xs2[1])),nnn.predict(np.array(xs2[2]))]
			ys2=[]
			for xs2_ in xs2:
				ys2.append(nnn.predict(np.array(xs2_)))


			#print(time.time()-start)
			
			# print()

			# Compare the two results
			# for i in range(len(ys2)):
			# 	if ys2[i].tolist()!=ys2b[i].tolist():
			# 		print(ys2[i].tolist())
			# 		print(ys2b[i].tolist())
			# 		print(xs2[i])
			# 		print("")

			#for i,key in enumerate(keys):
			for i,bkey in enumerate(xs2):
				k=bkey[-3:]
				ys=ys2[i].tolist()
				#ys = self.predict2(b+key)


				# punish=ys[:7].sum()/7
				# if punish<1:
				# 	punish=1
				# punish2=ys[8] #+ys[8]/10
				if prof == depth - 1:
					#heuristic = self.similar(ys[0], ys[6]) * self.similar(ys[1], ys[5]) * self.similar(ys[2], ys[4]) * ys[3] * sign(xs[8]) * ys[7]

					num_arms = param["lidar"]
					heuristic = 1 
					for i in range(num_arms // 2):
						heuristic *= self.similar(ys[i], ys[num_arms - 1 - i])
					# if is odd
					if num_arms % 2 == 1:
						heuristic *= ys[num_arms // 2] 
					heuristic *=  sign(ys[num_arms+1])
					if param["velocity"]==1:
						heuristic *= ys[num_arms]

					#heuristic = ys[8]
					if heuristic > best_heuristic:
						best_heuristic = heuristic
						r = k
				else:
					if prof == 0:
						base2.append((key, ys))
					else:
						base2.append((k, ys))
		#keys=[(0,0,1)]
			base = base2
		#print(best_heuristic)
		return (True if r2 == 1 else False for r2 in r)



	def usable(self):
		return hasattr(self, "model")

	def create(self,numxs,numys):
		self.numxs=numxs
		self.numys=numys
		import tensorflow as tf

		tf.config.set_visible_devices([], 'GPU')

		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense
		from tensorflow.python.client import device_lib
		from tensorflow.keras.optimizers import SGD, Adam

		#print(device_lib.list_local_devices())  # Verifica si detecta la GPU

		# np.random.seed(42)  
		# tf.random.set_seed(42)

		model = Sequential()
		model.add(Dense(1000, input_dim=numxs, activation='sigmoid',dtype='float16'))
		# model.add(Dense(7,dtype='float16'))
		# model.add(Dense(7,dtype='float16'))
		model.add(Dense(numys,dtype='float16'))
		#model.compile("adam", loss='mean_squared_error', metrics=['accuracy'])
		model.compile(optimizer=SGD(), loss='mean_squared_error')
		#model.compile(optimizer=Adam(learning_rate=1e-4, clipvalue=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
		

		self.model=model


	def add(self, xs, vel, heuristicDistance, leftb, rightb, upb):
		if self.old!=None:
			self.xs.append(self.old + [leftb, rightb, upb])
		self.old=xs + [vel, heuristicDistance]
		#self.xs.append(xs + [vel, heuristicDistance, leftb, rightb, upb])
		if len(self.xs) < self.initial_train+1:
			return
		if len(self.xs)> self.initial_train+1:
			return

		if not hasattr(self, "model"):
			self.create(len(xs)+5,len(xs)+2)

		tamTrain=self.initial_train
		model=self.model

		#X = np.random.rand(tamTrain, numxs)  
		X=np.array(self.xs[:tamTrain])
		# y = np.zeros((tamTrain, numys))  # Matriz para las salidas
		# for i in range(numys):
		# 	coeficientes = np.random.rand(numxs)  # Coeficientes aleatorios para cada característica
		# 	y[:, i] = np.dot(X, coeficientes) 
		y=np.array(self.xs[1:tamTrain+1])[:,0:self.numys]

		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
		model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)
		loss = model.evaluate(X_test, y_test)
		print(f'Pérdida en datos de prueba: {loss}')
		# import tensorflow as tf
		# converter = tf.lite.TFLiteConverter.from_keras_model(model)
		# converter.optimizations = [tf.lite.Optimize.DEFAULT]
		# tflite_model = converter.convert()
		# self.model=tflite_model

		#model.summary()
		#weights = model.get_weights()	




class Button:
	def __init__(self):
		# Initial button position
		self.button_position = None
		
		# Time since last click
		self.time_since_click = 0

	def predict(self, xs):
		"""
		Predict the output based on input features using a simple model.
		
		Parameters:
			xs (list): Input features
		
		Returns:
			list: Predicted outputs
		"""
		return 

	def add(self, xs, vel, leftb, rightb, upb):
		"""
		Add a new training sample to the button model.
		
		Parameters:
			xs (list): Input features
			vel (float): Velocity
			leftb (bool): Left button pressed
			rightb (bool): Right button pressed
			upb (bool): Up button pressed
		
		Returns:
			None
		"""
		if not self.button_position:
			self.button_position = xs
			return 
		left = 1 if leftb else 0
		right = 1 if rightb else 0
		up = 1 if upb else 0
		xt = self.button_position + [vel, left, right, up]
		yt = xs + [vel]
		yt = [int(y+1) for y in yt]

		ny = [[] for _ in range(len(yt))]

		for iy in range(len(yt)):
			x = [0]*len(xt)
			for t in range(1, 1000):
				for i in range(len(x)):
					if t % xt[i] == 0:
						x[i] += 1
				if t % yt[iy] == 0:
					ny[iy].append(x)
					x = [0]*len(xt)

		print(ny)
		# inicial
		# tiempo transcurrido desde el último disparo en t


class Car:
	def __init__(self,param, color=(255, 0, 0),color2=(255,0,255), position=[screen_width//2, screen_height//2], radius=50, velocity=[2, 2], image=None, friction=1.0):
		"""
		Initialize the car object.
		
		Parameters:
			color (tuple): Color of the car
			position (list): Initial position of the car
			radius (float): Radius of the car
			velocity (list): Initial velocity of the car
			image (str): Image file name (optional)
			friction (float): Friction coefficient
		
		Returns:
			None
		"""
		self.color = color
		self.color2=color2
		self.position = position.copy()
		self.radius = radius
		self.velocity = velocity.copy()
		self.motor = [0, 0]
		self.image = image
		self.friction = friction**3
		self.angle = math.pi
		self.left = False
		self.right = False
		self.giroVel = 1 / 180*3
		self.maxVelocity = 0.005*3*2*np.random.rand()
		self.p=param
		
		if self.p!=None:
			if self.p["type"]=="pursuit":
				self.nn = Pursuit()
			elif self.p["type"]=="regression":
				self.nn = Regressor()
			else:
				self.nn= NeuralNetwork()

			self.nn.initial_train=self.p["initial_train"]*1000
			#self.nn=Pulsante()

		self.rounds = 0

	def setPosition(self, i, total):
		"""
		Set the position of the car. Count the number of circuits completed.

		Parameters:
			i (int): Index of the circuit
			total (int): Total number of circuits
		"""
		if not hasattr(self, "initialSegment"):
			self.initialSegment = i
			self.expected = (i+1)%total
		
		if self.expected == i:
			self.rounds += 1/total
			self.expected = (i+1)%total

	def drawHand(self,pygame,screen, left, right, up, learn,cars, *circuit):
		"""
		Draw the hand of the car.
		
		Parameters:
			left (bool): Left button pressed
			right (bool): Right button pressed
			up (bool): Up button pressed
			learn (bool): Learn from data
			circuit (list): List of circuits
		
		Returns:
			list: Predicted outputs
		"""
		q0 = self.position

		arm_size=self.p["arm_size"]
		num_arms = self.p["lidar"]

		view = [arm_size] * num_arms

		incAngulo = [0] * num_arms
		increment = math.pi / (num_arms-1) 
		for i in range(num_arms):
			incAngulo[i] = -math.pi / 2 + i * increment



		xs = []
		#punish=math.inf
		for k, inc in enumerate(incAngulo):
			v = view[k]
			angle = self.angle + inc
			q1 = (self.position[0] + math.sin(-angle) * v, self.position[1] + math.cos(-angle) * v)
			pygame.draw.line(screen, (0, 0, 255), q0, q1, 5)
			min = distance(subtract(q0, q1))

			# intercetion with the other car
			for car in cars:
				if car!=self:
					# point in segment (q0,q1) where car position is near
					pclosest=line_circle_intersection(q0,q1,car.position,car.radius)
					if pclosest:
						cand=distance(subtract(q0, pclosest))
						if cand<min:
							pygame.draw.circle(screen, (0, 255, 0), (int(pclosest[0]), int(pclosest[1])), 10)
							min=cand
						
			for c in circuit:
				for i in range(len(c.points)):
					p0 = c.points[i]
					p1 = c.points[(i+1)%len(c.points)]

					i = intersection(q0, q1, p0, p1)
					if i:
						#print(cand)
						cand = distance(subtract(q0, i))
						if cand < min:
							pygame.draw.circle(screen, (255, 0, 0), (int(i[0]), int(i[1])), 10)
							min = cand

			


			min=min/view[k]
			xs.append(min)
			# if min<punish:
			# 	punish=min
			#punish+=min
		vel = distance(self.velocity)
		# punish=1
		# if self.punish:
		#     if distance(self.punish)>0:
		#         punish=0
		if learn:
			self.nn.add(xs, vel, self.heuristicDistance , left, right, up)
		return xs + [vel,self.heuristicDistance]

	def thinkAndAct(self):
		"""
		Think and act based on the current situation.
		
		Returns:
			None
		"""
		self.moveUp(self.maxVelocity)
		self.aWhereToGo()

		if self.left:
			self.turnLeft()

		if self.right:
			self.turnRight()

	def turnRight(self):
		"""
		Turn right by a small angle.
		
		Returns:
			None
		"""
		self.angle += math.pi * self.giroVel
		self.moveUp()
		#ball.velocity[0] += 1

	def turnLeft(self):
		"""
		Turn left by a small angle.
		
		Returns:
			None
		"""
		self.angle -= math.pi * self.giroVel
		self.moveUp()
		#ball.velocity[0] -= 1

	def down(self):
		"""
		Decrease the motor speed by 10%.
		
		Returns:
			None
		"""
		self.motor = scalarMultiplication(0.9, self.motor)

	def aWhereToGo(self):
		"""
		Decide where to go based on the current situation.
		
		Returns:
			None
		"""
		ball = self
		if not hasattr(ball, "ext") or not hasattr(ball, "int"):
			return
		target = scalarMultiplication(0.5, sum(ball.ext, ball.int))
		self.target = target
		
		vdir = subtract(target, ball.position)
		#if distance(vdir)>10:
		a = -math.atan2(vdir[0], vdir[1])
		while a < 0:
			a += 2 * math.pi
		while a > 2 * math.pi:
			a -= 2 * math.pi
		while ball.angle < 0:
			ball.angle += 2 * math.pi
		while ball.angle > 2 * math.pi:
			ball.angle -= 2 * math.pi

		dif = abs(a - ball.angle)
		#if dif>math.pi/10:
		self.left = False
		self.right = False
		if dif < math.pi:
			if a > ball.angle:
				self.right = True
			if a < ball.angle:
				self.left = True
		else:
			if a > ball.angle:
				self.left = True
			if a < ball.angle:
				self.right = True

	def moveUp(self, vel=0.0):
		"""
		Move the car up by a certain velocity.
		
		Parameters:
			vel (float): Velocity
		
		Returns:
			None
		"""
		vel = distance(self.motor) + vel
		self.motor[0] = math.sin(-self.angle) * vel
		self.motor[1] = math.cos(-self.angle) * vel

	def update(self):
		"""
		Update the car's position and velocity.
		
		Returns:
			None
		"""
		# vel=distance(self.velocity)
		# self.velocity[0] = math.sin(-self.angle)*vel
		# self.velocity[1] = math.cos(-self.angle)*vel

		if hasattr(self, "ext") and hasattr(self, "int"):
			target = scalarMultiplication(0.5, sum(self.ext, self.int))
			initial = distance(subtract(self.position, target)) 


		self.position[0] += self.velocity[0]+self.motor[0]
		self.position[1] += self.velocity[1]+self.motor[1]

		if self.position[0] <= -self.radius:
			self.position[0] += screen_width+2*self.radius
		if screen_width+self.radius <= self.position[0]:
			self.position[0] -= screen_width+2*self.radius

		# if self.position[1] <= self.radius or self.position[1] >= screen_height - self.radius:
		# 	self.velocity[1] = -self.velocity[1]

		if self.position[1] <= -self.radius:
			self.position[1] += screen_height + 2 * self.radius
		if screen_height + self.radius <= self.position[1]:
			self.position[1] -= screen_height + 2 * self.radius

		# if self.position[1] <= self.radius:
		# 	self.position[1] = self.radius

		# if screen_height - self.radius <= self.position[1]:
		# 	self.position[1] = screen_height - self.radius

		if hasattr(self, "ext") and hasattr(self, "int"):
			final=distance(subtract(self.position, target))
			self.heuristicDistance=initial-final
		else:
			self.heuristicDistance=0

		# 1 I have trouble taking corners because my car skids a lot, the steering wheel doesn't turn properly.
		# 0 I'm running through the walls.
		trasferencia=0.05
		
		self.velocity[0] = self.velocity[0]*self.friction+self.motor[0]*trasferencia
		self.velocity[1] = self.velocity[1]*self.friction+self.motor[1]*trasferencia
		self.motor[0]=self.motor[0]*(1-trasferencia)
		self.motor[1]=self.motor[1]*(1-trasferencia)

	def draw(self, pygame,screen):
		"""
		Draw the car on the screen.
		
		Parameters:
			pygame (object): Pygame object
		
		Returns:
			None
		"""
		# if self.image:
		# 	image = pygame.image.load(self.image)
		# 	image = pygame.transform.scale(image, (self.radius*2, self.radius*2))
		# 	screen.blit(image, (self.position[0]-self.radius, self.position[1]-self.radius))
		# else:
		# 	pygame.draw.circle(screen, self.color, self.position, self.radius)

		proporcion = math.pi / 6
		d=(self.radius,0)
		d0=rotate(d, math.pi/2-proporcion+self.angle)
		d05=rotate(d, math.pi/2+self.angle)
		d1=rotate(d, math.pi/2+proporcion+self.angle)
		d2=rotate(d, math.pi*3/2-proporcion+self.angle)
		d3=rotate(d, math.pi*3/2+proporcion+self.angle)
		d0=sum(d0,self.position)
		d05=sum(d05,self.position)
		d1=sum(d1,self.position)
		d2=sum(d2,self.position)
		d3=sum(d3,self.position)
		if self.nn.usable():
			pygame.draw.polygon(screen, self.color2, [d0,d05,d1,d2,d3])
		else:
			pygame.draw.polygon(screen, self.color, [d0,d05,d1,d2,d3])

		# draw circle in target
		if hasattr(self, "target"):
			pygame.draw.circle(screen, self.color, (int(self.target[0]), int(self.target[1])), 10)
		# pygame.draw.line(screen, self.color, d0, d1)
		# pygame.draw.line(screen, self.color, d1, d2)
		# pygame.draw.line(screen, self.color, d2, d3)
		# pygame.draw.line(screen, self.color, d3, d0)

		# draw text with the number of rounds 
		font = pygame.font.Font(None, 20)
		text = font.render(str(round(self.rounds,1)), True, self.color)
		# text color is self.color
		
		screen.blit(text, (self.position[0]-10, self.position[1]+15))
		
	def resetEnergy(self):
		"""
		Reset the car's energy.
		
		Returns:
			None
		"""
		self.energy = [0, 0]

	def transferEnergy(self, energy, p):
		"""
		Transfer energy from one car to another.
		
		Parameters:
			energy (float): Energy to transfer
			p (Car): Other car
		
		Returns:
			None
		"""
		# measure the distance between the two cars
		# if the distance is less than the sum of the radii of the two cars
		# then transfer energy multiplied by 10
		dis = distance(subtract(self.position, p.position))
		if dis < self.radius + p.radius:
			energy =energy+0.0001+energy*(1-dis/(self.radius + p.radius))
		auxX= energy*(p.velocity[0]*p.radius)
		self.energy[0] +=auxX
		p.energy[0] -=auxX
		auxY= energy*(p.velocity[1]*p.radius)
		self.energy[1] +=auxY
		p.energy[1] -=auxY

	def applyEnergy(self):
		"""
		Apply the car's energy to its velocity.
		
		Returns:
			None
		"""
		self.velocity[0] += self.energy[0]/self.radius
		self.velocity[1] += self.energy[1]/self.radius


class Circuit:
	def __init__(self, ext=None):
		"""
		Initialize the circuit object.
		
		Parameters:
			ext (Circuit): External circuit (optional)
		
		Returns:
			None
		"""
		self.points = []
		self.ext=ext

	def add(self, point):
		"""
		Add a new point to the circuit.
		
		Parameters:
			point (tuple): New point
		
		Returns:
			None
		"""
		self.points.append(point)

		print(self.points)

		if self.ext:
			if 2<len(self.points):
				ps1=[]
				
				for i in range(len(self.points)):
					p0=self.points[i-1]
					p1=self.points[i]

					d=vNormalizedSize(p0,p1,100)
					d2=rotate(d,-math.pi/2)
					d3=sum(d2,p0)
					ps1.append(d3)

					d=vNormalizedSize(p1,p0,100)
					d2=rotate(d,math.pi/2)
					d3=sum(d2,p1)
					ps1.append(d3)

				self.ext.points=[]
				for i in range(0,len(ps1),2):
					p0=ps1[i-2]
					p1=ps1[i-1]
					p2=ps1[i]
					p3=ps1[(i+1)%len(ps1)]

					pi=find_intersection(p0,p1,p2,p3)
					if pi:
						self.ext.points.append(pi)


	
	def draw(self, pygame, screen):
		"""
		Draw the circuit on the screen.
		
		Parameters:
			pygame (object): Pygame object
		
		Returns:
			None
		"""
		if len(self.points)<=1:
			return
		for i in range(len(self.points)-1):
			pygame.draw.line(screen, (0,255,0), self.points[i], self.points[i+1], 5)		
		pygame.draw.line(screen, (0,255,0), self.points[len(self.points)-1], self.points[0], 5)

	def collision(self, pygame, ps):
		"""
		Check for collisions between cars and the circuit.
		
		Parameters:
			pygame (object): Pygame object
			ps (list): List of cars
		
		Returns:
			None
		"""
		if len(self.points)<=3:
			return
		
		total=len(self.points)
		
		for p in ps:
			closest = math.inf
			memory=None
			#p.punish=None

			for i in range(total):
				s0=self.points[i]
				if i==len(self.points)-1:
					s1=self.points[0]
					s2=self.points[1]
				else:
					s1=self.points[i+1]
					if i==len(self.points)-2:
						s2=self.points[0]
					else:
						s2=self.points[i+2]
			
				d1=subtract(s1,s0)
				d2=subtract(p.position,s0)
				hasta=distance(d1)
				t3=scalarProduct(d1,d2)/hasta

				valido=0<=t3 and t3<=hasta
				cand=0
				if valido:
					d4=scalarMultiplication(t3,normalizedVector(d1))
					cand=distance(subtract(d2,d4))
					d5=sum(s0,d4)
				else:
					valido=True
					ds0=distance(subtract(p.position,s0))
					ds1=distance(subtract(p.position,s1))
					#ds2=distance(subtract(p.position,s2))
					cand=min(ds0,ds1)
					if ds0<ds1:
						d5=s0
					else:
						d5=s1
					
				if valido:
					if cand<closest:
						zpv=zVectorProduct(d1,d2)
						proyection=p.p["proyection"]/10
						memory=(d5,zpv,sum(s1,scalarMultiplication(proyection, subtract(s2,s1))),i)
						closest=cand
			
			if memory:
				d5,zpv,s1,i=memory

				if not self.ext: # Is exterior
					zpv=-zpv
					p.ext=s1
				else:
					p.int=s1

				if zpv>0:
					#pygame.draw.circle(screen, (255,0,0), d5,10)
					pass
				else:
					p.setPosition(i,total)
					directionalForce=subtract(p.position,d5)
					force=p.radius-distance(directionalForce)
					if force>0:
						den=normalizedVector(directionalForce)
						vforce=scalarMultiplication(force,den)

						inicial=distance(p.velocity)

						p.velocity[0] =0
						p.velocity[1] =0

						p.velocity[0] += vforce[0]
						p.velocity[1] += vforce[1]

						p.velocity=vNormalizedSize((0,0),p.velocity,inicial)
						#p.punish=vforce

					#pygame.draw.circle(screen, (0,255,0), d5,10)

def line_circle_intersection(p0, p1, e, r):
    # Convert points to numpy arrays
    p0 = np.array(p0)
    p1 = np.array(p1)
    e = np.array(e)
    
    # Vector along the line (p1 - p0)
    d = p1 - p0
    # Vector from the center of the circle to p0
    f = p0 - e
    
    # Coefficients for the quadratic equation
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2
    
    # Solve the quadratic equation: at^2 + bt + c = 0
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        # No intersection
        return None
    elif discriminant == 0:
        # One intersection (tangent line)
        t = -b / (2 * a)
        intersection = p0 + t * d
        return tuple(intersection)
    else:
		# Two intersections
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # Filter intersections that lie within the segment [p0, p1]
        valid_intersections = []
        
        if 0 <= t1 <= 1:
            intersection1 = p0 + t1 * d
            valid_intersections.append((intersection1, t1))
        
        if 0 <= t2 <= 1:
            intersection2 = p0 + t2 * d
            valid_intersections.append((intersection2, t2))
        
        # Check if there are valid intersections on the segment
        if valid_intersections:
            # Find the intersection closest to p0
            closest_intersection = min(valid_intersections, key=lambda x: x[1])  # Sort by t value
            return tuple(closest_intersection[0])
        else:
            return None


def sum(v1,v2):
	"""
	Sum two vectors.
	
	Parameters:
		v1 (list): First vector
		v2 (list): Second vector
	
	Returns:
		list: Sum of the two vectors
	"""
	return [v1[0]+v2[0],v1[1]+v2[1]]

def scalarMultiplication(a,v):
	"""
	Multiply a scalar by a vector.
	
	Parameters:
		a (float): Scalar
		v (list): Vector
	
	Returns:
		list: Product of the scalar and the vector
	"""
	return [a*v[0],a*v[1]]

def subtract(v1,v2):
	"""
	Subtract two vectors.
	
	Parameters:
		v1 (list): First vector
		v2 (list): Second vector
	
	Returns:
		list: Difference of the two vectors
	"""
	return [v1[0]-v2[0],v1[1]-v2[1]]

def scalarProduct(v1,v2):
	"""
	Compute the dot product of two vectors.
	
	Parameters:
		v1 (list): First vector
		v2 (list): Second vector
	
	Returns:
		float: Dot product of the two vectors
	"""
	return v1[0]*v2[0]+v1[1]*v2[1]

def distance(v):
	"""
	Compute the distance between two points.
	
	Parameters:
		v (list): Vector
	
	Returns:
		float: Distance of the vector
	"""
	return (v[0]**2+v[1]**2)**0.5

def normalizedVector(v):
	"""
	Normalize a vector.
	
	Parameters:
		v (list): Vector
	
	Returns:
		list: Normalized vector
	"""
	d=distance(v)
	return [v[0]/d,v[1]/d]

def zVectorProduct(v1,v2):
	"""
	Compute the cross product of two vectors.
	
	Parameters:
		v1 (list): First vector
		v2 (list): Second vector
	
	Returns:
		float: Cross product of the two vectors
	"""
	return v1[0]*v2[1]-v1[1]*v2[0]

def vNormalizedSize(p0,p1,Siz):
	"""
	Compute a vector with a given magnitude and direction.
	
	Parameters:
		p0 (tuple): First point
		p1 (tuple): Second point
		Siz (float): Magnitude of the vector
	
	Returns:
		list: Vector with the given magnitude and direction
	"""
	d=subtract(p1,p0)
	mul=Siz/distance(d)
	return [d[0]*mul,d[1]*mul]

def rotate(v,angulo):
	"""
	Rotate a vector by a certain angle.
	
	Parameters:
		v (list): Vector
		angulo (float): Angle of rotation
	
	Returns:
		list: Rotated vector
	"""
	return [v[0]*math.cos(angulo)-v[1]*math.sin(angulo),v[0]*math.sin(angulo)+v[1]*math.cos(angulo)]

def intersection(p0,p1,p2,p3):
	# Verify is rectangle r1 overlap with r2
	# AABB: Axis-Aligned Bounding Box
	maxAx = max(p0[0], p1[0])
	minAx = min(p0[0], p1[0])
	maxBx = max(p2[0], p3[0])
	minBx = min(p2[0], p3[0])
	if maxAx < minBx or maxBx < minAx:
		return None
	maxAy = max(p0[1], p1[1])
	minAy = min(p0[1], p1[1])
	maxBy = max(p2[1], p3[1])
	minBy = min(p2[1], p3[1])
	if maxAy < minBy or maxBy < minAy:
		return None
	i=find_intersection(p0, p1, p2, p3)
	# is i inside p2p3?
	if min(p2[0],p3[0])<=i[0]<=max(p2[0],p3[0]) and min(p2[1],p3[1])<=i[1]<=max(p2[1],p3[1]):
		if min(p0[0],p1[0])<=i[0]<=max(p0[0],p1[0]) and min(p0[1],p1[1])<=i[1]<=max(p0[1],p1[1]):
			return i
	return None

def upgrade(grade,val):
	return val
	val=list(val)
	val.append(1)
	r = val
	for _ in range(grade-1):
		r2=[]
		for v0 in r:
			for i, v1 in enumerate(val):
				r2.append(v0*v1)
				#versinable
				# for j in range(i, len(val)):
				# 	v2 = val[j]
				# 	for k in range(j,len(val)):
				# 		v3=val[k]
				# 		#r.append(v1*v2*v3)
				# 		for l in range(k,len(val)):
				# 			v4=val[l]
				# 			r.append(v1*v2*v3*v4)
				# # 	# r.append(v1*v2)
		r=r2
	return np.array(r)

def find_intersection(p0, p1, p2, p3):
	"""
	Find the intersection point of two lines.
	
	Parameters:
		p0 (tuple): First point of the first line
		p1 (tuple): Second point of the first line
		p2 (tuple): First point of the second line
		p3 (tuple): Second point of the second line
	
	Returns:
		tuple: Intersection point or None if no intersection is found
	"""
	# Calculate the slopes of lines r1 and r2
	m1 = (p1[1] - p0[1]) / (p1[0] - p0[0]) if p1[0] != p0[0] else float('inf')
	m2 = (p3[1] - p2[1]) / (p3[0] - p2[0]) if p3[0] != p2[0] else float('inf')

	# Calculate the intercepts of lines r1 and r2
	b1 = p0[1] - m1 * p0[0]
	b2 = p2[1] - m2 * p2[0]

	# Verifica si las rectas son paralelas
	if m1 == m2:
		return None  # Las rectas son paralelas o coincidentes, no hay intersección única

	# Calculate the midpoint of the intersection
	if m1 == float('inf'):  # r1 is vertical
		x = p0[0]
		y = m2 * x + b2
	elif m2 == float('inf'):  # r2 is vertical
		x = p2[0]
		y = m1 * x + b1
	else:
		x = (b2 - b1) / (m1 - m2)
		y = m1 * x + b1

	return (x, y)

class Parameters:
	def __init__(self):
		self.parameters = {
			"type": ["pursuit","regression", "neural_network"],
			"proyection": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
			"velocity": [0,1],
			"lidar":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
			"view_cars":[0,1],
			"initial_train":[1,2,3,4,5,6,7,8],
			"arm_size":[50,100,200],
		}

	def createRandomCar(self,type):
		"""
		Create a random car.
		
		Returns:
			Car: Random car
		"""
		car={}
		name=""
		for key in self.parameters:
			if key=="type":
				car[key]=type
				continue
			car[key]=random.choice(self.parameters[key])	
			if isinstance(car[key],int):
				name+=key[0]+str(car[key])
			else:
				name+=key[0]+car[key][0]
		car["name"]=name

		return car
	
	def initializeNRandCars(self,n):
		"""
		Create n random cars.
		
		Parameters:
			n (int): Number of cars
		
		Returns:
			list: List of random cars
		"""
		self.cars=[]
		for i in range(n):
			if i<n/2:
				type="pursuit"
			else:
				if random.random()<0.5:
					type="regression"
				else:
					type="neural_network"
			car=self.createRandomCar(type)
			self.cars.append(car)

def localeCar(ps,pos):
	near=None
	distanceNear=math.inf
	for p in ps:
		d0=subtract(p.position,pos)
		d=distance(d0)
		if d<distanceNear:
			distanceNear=d
			near=p
	print(near.p["name"])
	for k,v in near.p.items():
		if k!="name":
			print(" ",k,v)

class VoidEvent:
	def __init__(self):
		pass

	def get(self):
		return []

class VoidKey:
	def __init__(self):
		pass

	def get_pressed(self):
		return {0:False}
	
class VoidDraw:
	def __init__(self):
		pass

	def circle(self,screen,color,position,radius):
		pass

	def line(self,screen,color,p0,p1,width):
		pass

	def polygon(self,screen,color,points):
		pass

class VoidFont:
	def __init__(self):
		pass

	def Font(self,font,size):
		return self
	
	def render(self,text,boolean,color):
		return None
	
	def init(self):
		pass

	def SysFont(self,font,size):
		return self

class VoidDisplay:
	def __init__(self):
		pass
	def set_mode(self,screen):
		return self
	
	def fill(self,color):
		pass

	def blit(self,text,position):
		pass

	def flip(self):
		pass

class VoidPygame:
	def __init__(self):
		self.event=VoidEvent()
		self.key=VoidKey()
		self.K_SPACE=0
		self.K_UP=0
		self.K_DOWN=0
		self.K_LEFT=0
		self.K_RIGHT=0
		self.draw=VoidDraw()
		self.font=VoidFont()
		self.display=VoidDisplay()

	def init(self):
		pass

	def quit(self):
		pass


def execute(parameters,pygame=None):	
	if pygame==None:
		pygame=VoidPygame()
	screen = pygame.display.set_mode((screen_width, screen_height))

	# Initialize Pygame
	pygame.init()

	circuitDesign=False
	ps=[]
	ball=Car(None,color=(255,0,0),image="yo.png",friction=0.99,radius=15)
	#ps.append(ball)
	for i in range(len(parameters.cars)):
		param=parameters.cars[i]
		
		if param["type"]=="pursuit":
			color=(0,255,0)
			color2=(0,255,255)
		elif param["type"]=="regression":
			color=(255,0,0)
			color2=(255,0,255)
		else:
			color=(255,255,0)
			color2=(255,255,255)
		car=Car(param,friction=0.99,color=color,color2=color2,position=[random.randint(0,screen_width),random.randint(0,screen_height)],radius=random.randint(10,20),velocity=[0,0])
		car.radius=10
		if param["type"]=="pursuit":
			car.maxVelocity=0.005*6
		else:
			car.maxVelocity=0.005*12

		car.hand=None
		ps.append(car)

	circuitExt=Circuit()
	circuit=Circuit(circuitExt)

	for p in [(180, 551), (174, 483), (170, 432), (175, 380), (192, 296), (220, 242), (247, 218), (309, 219), (403, 196), (476, 159), (547, 168), (617, 194), (695, 238), (766, 234), (823, 192), (872, 155), (948, 131), (1037, 125), (1152, 143), (1226, 182), (1286, 240), (1339, 325), (1334, 387), (1259, 442), (1154, 384), (1052, 304), (900, 333), (821, 501), (847, 616), (934, 695), (1018, 714), (1148, 733), (1355, 758), (1395, 793), (1400, 821), (1382, 853), (1325, 881), (1012, 839), (809, 860), (684, 728), (711, 603), (733, 382), (632, 275), (355, 315), (296, 430), (415, 540), (424, 618), (421, 775), (332, 843), (183, 831), (142, 753)]:
		circuit.add(p)
	# Main loop game

	hand=None
	running = True

	inicializeVideo=time.time()
	frame=0

	while running:
		

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if circuitDesign:
					circuit.add(event.pos)
				else:
					localeCar(ps,event.pos)

		keys = pygame.key.get_pressed()

		screen.fill(black)

		frame+=1
		if frame>10:
			fpstime=time.time()-inicializeVideo
			fps=frame/fpstime
			# Draw on screen fps
			pygame.font.init()
			myfont = pygame.font.SysFont('Comic Sans MS', 20)
			textsurface = myfont.render(str(int(fps))+" FPS", False, (255, 0, 0))
			screen.blit(textsurface,(0,0))

		# Move the ball with the arrow keys
		# ball.velocity[0] /=1.001
		# ball.velocity[1] /=1.001

		ball.left=False
		ball.right=False
		ball.up=False
		space=False

		for p in ps:
			if not p.nn.usable():
				p.thinkAndAct()
				p.up=True
			else:
				p.left,p.right,p.up=p.nn.predict(p.hand,p.p)
				if p.up:
					p.moveUp(p.maxVelocity)
				if p.left:
					p.turnLeft()
				if p.right:
					p.turnRight()


		if keys[pygame.K_SPACE] or space:
			ball.left,ball.right,ball.up=ball.nn.predict(hand)
		
		
		if keys[pygame.K_UP] or ball.up:
			ball.moveUp(ball.maxVelocity)
			ball.up=True
			#ball.aWhereToGo()

		if keys[pygame.K_DOWN]:
			ball.down()		

		if keys[pygame.K_LEFT] or ball.left:
			ball.turnLeft()
			ball.left=True

			
		if keys[pygame.K_RIGHT] or ball.right:
			ball.turnRight()
			ball.right=True

		# for p in ps:
		# 	if p!=ball:
		# 		p.thinkAndAct()
		
		for p in ps:
			p.update()

		#print(ball.heuristicDistance)

		for p1 in ps:
			p1.resetEnergy()

		# collision between cars
		for p1 in ps:
			for p2 in ps:
				if p1 != p2:
					if (p1.position[0]-p2.position[0])**2+(p1.position[1]-p2.position[1])**2 < (p1.radius+p2.radius)**2:
						p1.transferEnergy(1,p2)
						# p1.velocity[0] = -p1.velocity[0]
						# p1.velocity[1] = -p1.velocity[1]
		for p1 in ps:
			p1.applyEnergy()

		learn=not keys[pygame.K_SPACE] and not space
		for p in ps:
			if p.p["type"]!="pursuit":
				p.hand=p.drawHand(pygame, screen, p.left,p.right,p.up,learn,ps if p.p["view_cars"]==1 else [], circuit,circuitExt)

		circuit.draw(pygame,screen)
		circuitExt.draw(pygame,screen)
		for p in ps:
			p.draw(pygame,screen)

		circuit.collision(pygame,ps)
		circuitExt.collision(pygame,ps)

		#learn=True

		#hand=ball.drawHand(ball.left,ball.right,ball.up,learn, circuit,circuitExt)


		# Update the display
		pygame.display.flip()

		# Control the game speed
		#pygame.time.delay(2)

		# detect end of race
		end=False
		for p in ps:
			if p.rounds>10:
				end=True
		if end:
			break

	# Print clasification
	for p in ps:
		p.p["rounds"]=p.rounds

	# Finish Pygame and close the window
	pygame.quit()
	#sys.exit()
	return p

if __name__ == '__main__':
	parameters=Parameters()
	parameters.initializeNRandCars(20)

	import pygame

	start=time.time()
	execute(parameters,pygame)
	print("Time:",time.time()-start)